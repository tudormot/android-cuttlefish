// Copyright 2026, The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use std::io::Result as IoResult;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::PathBuf;
use std::os::fd::AsFd;
use std::os::fd::BorrowedFd;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use v4l2r::PixelFormat;
use v4l2r::QueueType;
use v4l2r::bindings;
use v4l2r::bindings::v4l2_fmtdesc;
use v4l2r::bindings::v4l2_format;
use v4l2r::bindings::v4l2_requestbuffers;
use v4l2r::ioctl::BufferCapabilities;
use v4l2r::ioctl::BufferField;
use v4l2r::ioctl::BufferFlags;
use v4l2r::ioctl::CtrlId;
use v4l2r::ioctl::CtrlWhich;
use v4l2r::ioctl::EventType as V4l2EventType;
use v4l2r::ioctl::QueryCtrlFlags;
use v4l2r::ioctl::SubscribeEventFlags;
use v4l2r::ioctl::V4l2Buffer;
use v4l2r::ioctl::V4l2PlanesWithBackingMut;
use v4l2r::memory::MemoryType;
use virtio_media::VirtioMediaDevice;
use virtio_media::VirtioMediaDeviceSession;
use virtio_media::VirtioMediaEventQueue;
use virtio_media::VirtioMediaHostMemoryMapper;
use virtio_media::io::ReadFromDescriptorChain;
use virtio_media::io::WriteToDescriptorChain;
use virtio_media::ioctl::IoctlResult;
use virtio_media::ioctl::VirtioMediaIoctlHandler;
use virtio_media::ioctl::virtio_media_dispatch_ioctl;
use virtio_media::memfd::MemFdBuffer;
use virtio_media::mmap::MmapMappingManager;
use virtio_media::protocol::DequeueBufferEvent;
use virtio_media::protocol::SessionEvent;
use virtio_media::protocol::SgEntry;
use virtio_media::protocol::V4l2Event;
use virtio_media::protocol::V4l2Ioctl;
use virtio_media::protocol::VIRTIO_MEDIA_MMAP_FLAG_RW;
use std::str::FromStr;

/// https://developer.android.com/reference/android/hardware/camera2/CameraMetadata#LENS_FACING_FRONT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensFacing {
    Front = 0,
    Back = 1,
    External = 2,
}

impl FromStr for LensFacing {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "FRONT" => Ok(LensFacing::Front),
            "BACK" => Ok(LensFacing::Back),
            "EXTERNAL" => Ok(LensFacing::External),
            _ => Err(format!("Invalid lens facing: {}. Expected FRONT, BACK, or EXTERNAL", s)),
        }
    }
}

/// Current status of a buffer.
#[derive(Debug, PartialEq, Eq)]
enum BufferState {
    /// Buffer has just been created (or streamed off) and not been used yet.
    New,
    /// Buffer has been QBUF'd by the driver but not yet processed.
    Incoming,
    /// Buffer has been processed and is ready for dequeue.
    Outgoing {
        /// Sequence of the generated frame.
        sequence: u32,
    },
}

/// Information about a single buffer.
struct Buffer {
    /// Current state of the buffer.
    state: BufferState,
    /// V4L2 representation of this buffer to be sent to the guest when requested.
    v4l2_buffer: V4l2Buffer,
    /// Backing storage for the buffer.
    fd: MemFdBuffer,
    /// Offset that can be used to map the buffer.
    ///
    /// Cached from `v4l2_buffer` to avoid doing a match.
    offset: u32,
}

impl Buffer {
    fn new(v4l2_buffer: V4l2Buffer, fd: MemFdBuffer, offset: u32) -> Self {
        Self {
            state: BufferState::New,
            v4l2_buffer,
            fd,
            offset,
        }
    }

    fn unset_flag(flags: &mut BufferFlags, v: BufferFlags) {
        *flags &= !v;
    }

    /// Update the state of the buffer as well as its V4L2 representation.
    fn set_state(&mut self, state: BufferState) {
        let mut flags = self.v4l2_buffer.flags();
        match state {
            BufferState::New => {
                *self.v4l2_buffer.get_first_plane_mut().bytesused = 0;
                Self::unset_flag(&mut flags, BufferFlags::QUEUED);
            }
            BufferState::Incoming => {
                *self.v4l2_buffer.get_first_plane_mut().bytesused = 0;
                flags |= BufferFlags::QUEUED;
            }
            BufferState::Outgoing { sequence } => {
                self.v4l2_buffer.set_sequence(sequence);
                self.v4l2_buffer.set_timestamp(bindings::timeval {
                    tv_sec: (sequence + 1) as bindings::__time_t / 1000,
                    tv_usec: (sequence + 1) as bindings::__time_t % 1000,
                });
                Self::unset_flag(&mut flags, BufferFlags::QUEUED);
            }
        }
        self.v4l2_buffer.set_flags(flags);
        self.state = state;
    }
}

struct SessionSharedState {
    /// Id of the session.
    id: u32,
    /// Current iteration of the pattern generation cycle.
    iteration: u64,
    /// Buffers currently allocated for this session.
    buffers: Vec<Buffer>,
    /// FIFO of queued buffers awaiting processing.
    queued_buffers: VecDeque<usize>,
    /// Is the session currently streaming?
    streaming: bool,
}

/// Session data of [`EmulatedCamera`].
pub struct EmulatedCameraSession {
    shared: Arc<Mutex<SessionSharedState>>,
    /// Flag to signal the background reader thread to exit.
    exit_flag: Arc<AtomicBool>,
    /// Handle to join the background reader thread on session drop.
    reader_thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for EmulatedCameraSession {
    fn drop(&mut self) {
        log::info!("Dropping EmulatedCameraSession, stopping reader thread...");
        self.exit_flag.store(true, Ordering::SeqCst);
        if let Some(handle) = self.reader_thread.take() {
            let _ = handle.join();
        }
        log::info!("EmulatedCameraSession reader thread stopped successfully.");
    }
}

impl VirtioMediaDeviceSession for EmulatedCameraSession {
    fn poll_fd(&self) -> Option<BorrowedFd<'_>> {
        None
    }
}

impl EmulatedCameraSession {
    pub fn write_yuv420_pattern<W: std::io::Write>(iteration: u64, mut sink: W) -> std::io::Result<()> {
        let w = WIDTH as usize;
        let h = HEIGHT as usize;
        let y_size = w * h;
        let uv_size = y_size / 4;
        
        let mut y_bytes = vec![0u8; y_size];
        for y in 0..h {
            for x in 0..w {
                let val = ((x + y + (iteration as usize * 4)) % 256) as u8;
                y_bytes[y * w + x] = val;
            }
        }
        
        let mut u_bytes = vec![0u8; uv_size];
        let mut v_bytes = vec![0u8; uv_size];
        let uw = w / 2;
        let uh = h / 2;
        for y in 0..uh {
            for x in 0..uw {
                let u_val = ((x * 8 + (iteration as usize * 2)) % 256) as u8;
                let v_val = ((y * 8 + (iteration as usize * 3)) % 256) as u8;
                u_bytes[y * uw + x] = u_val;
                v_bytes[y * uw + x] = v_val;
            }
        }
        
        sink.write_all(&y_bytes)?;
        sink.write_all(&u_bytes)?;
        sink.write_all(&v_bytes)?;
        Ok(())
    }

    pub fn read_and_write_yuv_frame<R: Read, W: Write>(source: &mut R, mut sink: W) -> std::io::Result<()> {
        let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
        let mut frame_data = vec![0u8; frame_size];
        source.read_exact(&mut frame_data)?;
        sink.write_all(&frame_data)?;
        Ok(())
    }
}

/// Emulated camera used for testing Android camera stack.
///
/// This implementation looks forward to have feature parity with existing Android Guest Emulated
/// camera https://cs.android.com/android/platform/superproject/main/+/main:hardware/google/camera/devices/EmulatedCamera/
pub struct EmulatedCamera<Q: VirtioMediaEventQueue, HM: VirtioMediaHostMemoryMapper> {
    /// Queue used to send events to the guest.
    evt_queue: Q,
    /// Host MMAP mapping manager.
    mmap_manager: MmapMappingManager<HM>,
    /// ID of the session with allocated buffers, if any.
    ///
    /// v4l2-compliance checks that only a single session can have allocated buffers at a given
    /// time, since that's how actual hardware works - no two sessions can access a camera at the
    /// same time. It will fails if we allow simultaneous sessions to be active, so we need this
    /// artificial limitation to make it pass fully.
    active_session: Option<u32>,
    /// Lens facing configuration.
    lens_facing: LensFacing,
    /// Path to the Named Pipe for camera streaming.
    camera_pipe: Option<PathBuf>,
}

impl<Q, HM> EmulatedCamera<Q, HM>
where
    Q: VirtioMediaEventQueue,
    HM: VirtioMediaHostMemoryMapper,
{
    pub fn new(evt_queue: Q, mapper: HM, lens_facing: LensFacing, camera_pipe: Option<PathBuf>) -> Self {
        Self {
            evt_queue,
            mmap_manager: MmapMappingManager::from(mapper),
            active_session: None,
            lens_facing,
            camera_pipe,
        }
    }

    fn lens_facing_query_ext_ctrl(&self) -> bindings::v4l2_query_ext_ctrl {
        let name_str = "LENS_FACING";
        let mut name = [0u8; 32];
        name[0..name_str.len()].copy_from_slice(name_str.as_bytes());
        bindings::v4l2_query_ext_ctrl {
            id: CID_LENS_FACING,
            type_: bindings::v4l2_ctrl_type_V4L2_CTRL_TYPE_INTEGER,
            name: name.map(|b| b as i8),
            minimum: LensFacing::Front as i64,
            maximum: LensFacing::External as i64,
            step: 1,
            default_value: self.lens_facing as i64,
            flags: bindings::V4L2_CTRL_FLAG_READ_ONLY,
            elems: 1,
            elem_size: std::mem::size_of::<u32>() as u32,
            ..Default::default()
        }
    }
}

impl<Q, HM, Reader, Writer> VirtioMediaDevice<Reader, Writer> for EmulatedCamera<Q, HM>
where
    Q: VirtioMediaEventQueue + Clone + Send + 'static,
    HM: VirtioMediaHostMemoryMapper,
    Reader: ReadFromDescriptorChain,
    Writer: WriteToDescriptorChain,
{
    type Session = EmulatedCameraSession;

    fn new_session(&mut self, session_id: u32) -> std::result::Result<Self::Session, i32> {
        let shared = Arc::new(Mutex::new(SessionSharedState {
            id: session_id,
            iteration: 0,
            buffers: Default::default(),
            queued_buffers: Default::default(),
            streaming: false,
        }));

        let exit_flag = Arc::new(AtomicBool::new(false));

        let shared_clone = shared.clone();
        let exit_flag_clone = exit_flag.clone();
        let mut evt_queue_clone = self.evt_queue.clone();
        let camera_pipe = self.camera_pipe.clone();

        let reader_thread = std::thread::spawn(move || {
            let mut pipe: Option<File> = None;
            let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
            let mut read_buffer = Vec::with_capacity(frame_size);

            log::info!("Background reader thread started for session {}", session_id);

            while !exit_flag_clone.load(Ordering::SeqCst) {
                let mut frame_read_successfully = false;

                if let Some(ref path) = camera_pipe {
                    let pipe_path = path.to_str().unwrap();
                    if pipe.is_none() {
                        use std::os::unix::fs::OpenOptionsExt;
                        match std::fs::OpenOptions::new()
                            .read(true)
                            .custom_flags(libc::O_NONBLOCK)
                            .open(pipe_path)
                        {
                            Ok(file) => {
                                pipe = Some(file);
                                read_buffer.clear();
                                log::info!("Reader thread successfully opened non-blocking pipe {}", pipe_path);
                            }
                            Err(_) => {
                                // Pipe not available, will fall back
                            }
                        }
                    }

                    if let Some(ref mut file) = pipe {
                        let needed = frame_size - read_buffer.len();
                        let mut temp_buf = vec![0u8; needed];

                        match file.read(&mut temp_buf) {
                            Ok(0) => {
                                log::warn!("Reader thread: pipe EOF detected (writer disconnected).");
                                pipe = None;
                                read_buffer.clear();
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                            Ok(n) => {
                                read_buffer.extend_from_slice(&temp_buf[..n]);
                                if read_buffer.len() == frame_size {
                                    let mut shared = shared_clone.lock().unwrap();
                                    if shared.streaming {
                                        if let Some(buf_id) = shared.queued_buffers.pop_front() {
                                            let iteration = shared.iteration;
                                            let id = shared.id;
                                            let mut success = false;
                                            {
                                                if let Some(buffer) = shared.buffers.get_mut(buf_id) {
                                                    if let Ok(_) = buffer.fd.as_file().seek(SeekFrom::Start(0)) {
                                                        if let Ok(_) = buffer.fd.as_file().write_all(&read_buffer) {
                                                            *buffer.v4l2_buffer.get_first_plane_mut().bytesused = BUFFER_SIZE;
                                                            buffer.set_state(BufferState::Outgoing {
                                                                sequence: iteration as u32,
                                                            });
                                                            evt_queue_clone.send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                                                                id,
                                                                buffer.v4l2_buffer.clone(),
                                                            )));
                                                            success = true;
                                                        }
                                                    }
                                                }
                                            }
                                            if success {
                                                shared.iteration += 1;
                                            }
                                        }
                                    }
                                    read_buffer.clear();
                                    frame_read_successfully = true;
                                }
                            }
                            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                // No data, will fall back
                            }
                            Err(e) => {
                                log::error!("Reader thread: pipe read error: {}", e);
                                pipe = None;
                                read_buffer.clear();
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                        }
                    }
                }

                if !frame_read_successfully {
                    let mut shared = shared_clone.lock().unwrap();
                    if shared.streaming {
                        if let Some(buf_id) = shared.queued_buffers.pop_front() {
                            let iteration = shared.iteration;
                            let id = shared.id;
                            let mut success = false;
                            {
                                if let Some(buffer) = shared.buffers.get_mut(buf_id) {
                                    if let Ok(_) = buffer.fd.as_file().seek(SeekFrom::Start(0)) {
                                        if let Ok(_) = EmulatedCameraSession::write_yuv420_pattern(iteration, buffer.fd.as_file()) {
                                            *buffer.v4l2_buffer.get_first_plane_mut().bytesused = BUFFER_SIZE;
                                            buffer.set_state(BufferState::Outgoing {
                                                sequence: iteration as u32,
                                            });
                                            evt_queue_clone.send_event(V4l2Event::DequeueBuffer(DequeueBufferEvent::new(
                                                id,
                                                buffer.v4l2_buffer.clone(),
                                            )));
                                            success = true;
                                        }
                                    }
                                }
                            }
                            if success {
                                shared.iteration += 1;
                            }
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(33));
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
            log::info!("Background reader thread exiting for session {}", session_id);
        });

        Ok(EmulatedCameraSession {
            shared,
            exit_flag,
            reader_thread: Some(reader_thread),
        })
    }

    fn close_session(&mut self, session: Self::Session) {
        let session_id = session.shared.lock().unwrap().id;
        if self.active_session != Some(session_id) {
            return;
        }

        self.active_session = None;

        let shared = session.shared.lock().unwrap();
        for buffer in &shared.buffers {
            self.mmap_manager.unregister_buffer(buffer.offset);
        }
    }

    fn do_ioctl(
        &mut self,
        session: &mut Self::Session,
        ioctl: V4l2Ioctl,
        reader: &mut Reader,
        writer: &mut Writer,
    ) -> IoResult<()> {
        virtio_media_dispatch_ioctl(self, session, ioctl, reader, writer)
    }

    fn do_mmap(
        &mut self,
        session: &mut Self::Session,
        flags: u32,
        offset: u32,
    ) -> std::result::Result<(u64, u64), i32> {
        let mut shared = session.shared.lock().unwrap();
        let buffer = shared
            .buffers
            .iter_mut()
            .find(|b| b.offset == offset)
            .ok_or(libc::EINVAL)?;
        let rw = (flags & VIRTIO_MEDIA_MMAP_FLAG_RW) != 0;
        let fd = buffer.fd.as_file().as_fd();
        let (guest_addr, size) = self
            .mmap_manager
            .create_mapping(offset, fd, rw)
            .map_err(|_| libc::EINVAL)?;
        Ok((guest_addr, size))
    }

    fn do_munmap(&mut self, guest_addr: u64) -> std::result::Result<(), i32> {
        let _ = self.mmap_manager.remove_mapping(guest_addr);
        Ok(())
    }
}

// Use an offset for virtio-media custom camera class control id values.
const CID_OFFSET: u32 = bindings::V4L2_CID_CAMERA_CLASS_BASE + 0x100;
const CID_LENS_FACING: u32 = CID_OFFSET + 1;

const FRAME_RATE: u32 = 30;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const BYTES_PER_LINE: u32 = WIDTH;

const PIXELFORMAT: u32 = PixelFormat::from_fourcc(b"YU12").to_u32();
const BUFFER_SIZE: u32 = WIDTH * HEIGHT * 3 / 2;

const INPUTS: [bindings::v4l2_input; 1] = [bindings::v4l2_input {
    index: 0,
    name: *b"Default\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
    type_: bindings::V4L2_INPUT_TYPE_CAMERA,
    ..unsafe { std::mem::zeroed() }
}];

fn default_fmtdesc(queue: QueueType) -> v4l2_fmtdesc {
    v4l2_fmtdesc {
        index: 0,
        type_: queue as u32,
        pixelformat: PIXELFORMAT,
        ..Default::default()
    }
}

fn default_fmt(queue: QueueType) -> v4l2_format {
    let pix = bindings::v4l2_pix_format {
        width: WIDTH,
        height: HEIGHT,
        pixelformat: PIXELFORMAT,
        field: bindings::v4l2_field_V4L2_FIELD_NONE,
        bytesperline: BYTES_PER_LINE,
        sizeimage: BUFFER_SIZE,
        colorspace: bindings::v4l2_colorspace_V4L2_COLORSPACE_SRGB,
        ..Default::default()
    };

    v4l2_format {
        type_: queue as u32,
        fmt: bindings::v4l2_format__bindgen_ty_1 { pix },
    }
}

/// Implementations of the ioctls required by a v4l2 CAPTURE device.
impl<Q, HM> VirtioMediaIoctlHandler for EmulatedCamera<Q, HM>
where
    Q: VirtioMediaEventQueue,
    HM: VirtioMediaHostMemoryMapper,
{
    type Session = EmulatedCameraSession;

    fn enum_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2_fmtdesc> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        if index > 0 {
            return Err(libc::EINVAL);
        }

        Ok(default_fmtdesc(queue))
    }

    fn g_fmt(&mut self, _session: &Self::Session, queue: QueueType) -> IoctlResult<v4l2_format> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }

        Ok(default_fmt(queue))
    }

    fn s_fmt(
        &mut self,
        _session: &mut Self::Session,
        queue: QueueType,
        _format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }

        Ok(default_fmt(queue))
    }

    fn try_fmt(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
        _format: v4l2_format,
    ) -> IoctlResult<v4l2_format> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }

        Ok(default_fmt(queue))
    }

    fn enum_framesizes(
        &mut self,
        _session: &Self::Session,
        index: u32,
        pixel_format: u32,
    ) -> IoctlResult<bindings::v4l2_frmsizeenum> {
        if index as usize > 0 {
            return Err(libc::EINVAL);
        }
        if pixel_format != PIXELFORMAT {
            return Err(libc::EINVAL);
        }

        Ok(bindings::v4l2_frmsizeenum {
            index,
            pixel_format,
            type_: bindings::v4l2_frmsizetypes_V4L2_FRMSIZE_TYPE_DISCRETE,
            __bindgen_anon_1: bindings::v4l2_frmsizeenum__bindgen_ty_1 {
                discrete: bindings::v4l2_frmsize_discrete {
                    width: WIDTH,
                    height: HEIGHT,
                },
            },
            ..Default::default()
        })
    }

    fn enum_frameintervals(
        &mut self,
        _session: &Self::Session,
        index: u32,
        pixel_format: u32,
        width: u32,
        height: u32,
    ) -> IoctlResult<bindings::v4l2_frmivalenum> {
        if index > 0 {
            return Err(libc::EINVAL);
        }
        if pixel_format != PIXELFORMAT {
            return Err(libc::EINVAL);
        }
        if width != WIDTH || height != HEIGHT {
            return Err(libc::EINVAL);
        }

        Ok(bindings::v4l2_frmivalenum {
            index,
            pixel_format,
            width,
            height,
            type_: bindings::v4l2_frmivaltypes_V4L2_FRMIVAL_TYPE_DISCRETE,
            __bindgen_anon_1: bindings::v4l2_frmivalenum__bindgen_ty_1 {
                discrete: bindings::v4l2_fract {
                    numerator: 1,
                    denominator: FRAME_RATE,
                },
            },
            ..Default::default()
        })
    }

    fn g_parm(
        &mut self,
        _session: &Self::Session,
        queue: QueueType,
    ) -> IoctlResult<bindings::v4l2_streamparm> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        };

        let mut parm = bindings::v4l2_streamparm {
            type_: queue as u32,
            ..Default::default()
        };

        // SAFETY: The `parm` union is used for the capture type.
        let capture = unsafe { &mut parm.parm.capture };
        capture.capability = bindings::V4L2_CAP_TIMEPERFRAME;
        capture.timeperframe = bindings::v4l2_fract {
            numerator: 1,
            denominator: FRAME_RATE,
        };

        Ok(parm)
    }

    fn s_parm(
        &mut self,
        _session: &mut Self::Session,
        mut parm: bindings::v4l2_streamparm,
    ) -> IoctlResult<bindings::v4l2_streamparm> {
        if parm.type_ != QueueType::VideoCapture as u32 {
            return Err(libc::EINVAL);
        }

        // We just return the fixed values, ignoring what the user set.
        // SAFETY: The `parm` union is used for the capture type.
        let capture = unsafe { &mut parm.parm.capture };
        capture.capability = bindings::V4L2_CAP_TIMEPERFRAME;
        capture.timeperframe = bindings::v4l2_fract {
            numerator: 1,
            denominator: FRAME_RATE,
        };

        Ok(parm)
    }

    fn reqbufs(
        &mut self,
        session: &mut Self::Session,
        queue: QueueType,
        memory: MemoryType,
        count: u32,
    ) -> IoctlResult<v4l2_requestbuffers> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        if memory != MemoryType::Mmap {
            return Err(libc::EINVAL);
        }
        
        let session_id = session.shared.lock().unwrap().id;
        let streaming = session.shared.lock().unwrap().streaming;

        if streaming {
            return Err(libc::EBUSY);
        }
        // Buffers cannot be requested on a session if there is already another session with
        // allocated buffers.
        match self.active_session {
            Some(id) if id != session_id => return Err(libc::EBUSY),
            _ => (),
        }

        // Reqbufs(0) is an implicit streamoff.
        if count == 0 {
            self.active_session = None;
            self.streamoff(session, queue)?;
        } else {
            // TODO factorize with streamoff.
            let mut shared = session.shared.lock().unwrap();
            shared.queued_buffers.clear();
            for buffer in shared.buffers.iter_mut() {
                buffer.set_state(BufferState::New);
            }
            self.active_session = Some(session_id);
        }

        let count = std::cmp::min(count, 32);

        {
            let shared = session.shared.lock().unwrap();
            for buffer in &shared.buffers {
                self.mmap_manager.unregister_buffer(buffer.offset);
            }
        }

        let buffers = (0..count)
            .map(|i| {
                MemFdBuffer::new(BUFFER_SIZE as u64)
                    .map_err(|e| {
                        log::error!("failed to allocate MMAP buffers: {:#}", e);
                        libc::ENOMEM
                    })
                    .and_then(|fd| {
                        let offset = self
                            .mmap_manager
                            .register_buffer(None, BUFFER_SIZE)
                            .map_err(|_| libc::EINVAL)?;

                        let mut v4l2_buffer =
                            V4l2Buffer::new(QueueType::VideoCapture, i, MemoryType::Mmap);
                        if let V4l2PlanesWithBackingMut::Mmap(mut planes) =
                            v4l2_buffer.planes_with_backing_iter_mut()
                        {
                            // SAFETY: every buffer has at least one plane.
                            let mut plane = planes.next().unwrap();
                            plane.set_mem_offset(offset);
                            *plane.length = BUFFER_SIZE;
                        } else {
                            // SAFETY: we have just set the buffer type to MMAP. Reaching this point means a bug in
                            // the code.
                            panic!()
                        }
                        v4l2_buffer.set_field(BufferField::None);
                        v4l2_buffer.set_flags(BufferFlags::TIMESTAMP_MONOTONIC);

                        Ok(Buffer::new(v4l2_buffer, fd, offset))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        session.shared.lock().unwrap().buffers = buffers;

        Ok(v4l2_requestbuffers {
            count,
            type_: queue as u32,
            memory: memory as u32,
            capabilities: (BufferCapabilities::SUPPORTS_MMAP
                | BufferCapabilities::SUPPORTS_ORPHANED_BUFS)
                .bits(),
            ..Default::default()
        })
    }

    fn querybuf(
        &mut self,
        session: &Self::Session,
        queue: QueueType,
        index: u32,
    ) -> IoctlResult<v4l2r::ioctl::V4l2Buffer> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        let shared = session.shared.lock().unwrap();
        let buffer = shared.buffers.get(index as usize).ok_or(libc::EINVAL)?;

        Ok(buffer.v4l2_buffer.clone())
    }

    fn qbuf(
        &mut self,
        session: &mut Self::Session,
        buffer: v4l2r::ioctl::V4l2Buffer,
        _guest_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<v4l2r::ioctl::V4l2Buffer> {
        let mut shared = session.shared.lock().unwrap();
        let buffer_index = buffer.index() as usize;
        {
            let host_buffer = shared
                .buffers
                .get_mut(buffer_index)
                .ok_or(libc::EINVAL)?;
            // Attempt to queue already queued buffer.
            if matches!(host_buffer.state, BufferState::Incoming) {
                return Err(libc::EINVAL);
            }

            host_buffer.set_state(BufferState::Incoming);
        }
        shared.queued_buffers.push_back(buffer_index);

        let buffer = shared.buffers[buffer_index].v4l2_buffer.clone();
        Ok(buffer)
    }

    fn streamon(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        let mut shared = session.shared.lock().unwrap();
        if shared.buffers.is_empty() {
            return Err(libc::EINVAL);
        }
        shared.streaming = true;

        Ok(())
    }

    fn streamoff(&mut self, session: &mut Self::Session, queue: QueueType) -> IoctlResult<()> {
        if queue != QueueType::VideoCapture {
            return Err(libc::EINVAL);
        }
        let mut shared = session.shared.lock().unwrap();
        shared.streaming = false;
        shared.queued_buffers.clear();
        for buffer in shared.buffers.iter_mut() {
            buffer.set_state(BufferState::New);
        }

        Ok(())
    }

    fn g_input(&mut self, _session: &Self::Session) -> IoctlResult<i32> {
        Ok(0)
    }

    fn s_input(&mut self, _session: &mut Self::Session, input: i32) -> IoctlResult<i32> {
        if input != 0 { Err(libc::EINVAL) } else { Ok(0) }
    }

    fn enuminput(
        &mut self,
        _session: &Self::Session,
        index: u32,
    ) -> IoctlResult<bindings::v4l2_input> {
        match INPUTS.get(index as usize) {
            Some(&input) => Ok(input),
            None => Err(libc::EINVAL),
        }
    }

    /// https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/vidioc-queryctrl.html#control-flags
    fn query_ext_ctrl(
        &mut self,
        _session: &Self::Session,
        id: CtrlId,
        flags: QueryCtrlFlags,
    ) -> IoctlResult<bindings::v4l2_query_ext_ctrl> {
        let id: u32 = unsafe { std::mem::transmute(id) };
        // If V4L2_CTRL_FLAG_NEXT_CTRL present returns the first control with a higher ID.
        if flags.contains(QueryCtrlFlags::NEXT) {
            if id < CID_LENS_FACING {
                return Ok(self.lens_facing_query_ext_ctrl());
            }
        } else if id == CID_LENS_FACING {
            return Ok(self.lens_facing_query_ext_ctrl());
        }
        return Err(libc::EINVAL);
    }

    fn g_ext_ctrls(
        &mut self,
        _session: &Self::Session,
        _which: CtrlWhich,
        ctrls: &mut bindings::v4l2_ext_controls,
        ctrl_array: &mut Vec<bindings::v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        for ctrl in ctrl_array {
            match ctrl.id {
                CID_LENS_FACING => {
                    ctrl.__bindgen_anon_1.value = self.lens_facing as i32;
                }
                _ => {
                    ctrls.error_idx = ctrls.count;
                    return Err(libc::EINVAL);
                }
            }
        }
        Ok(())
    }

    fn try_ext_ctrls(
        &mut self,
        _session: &Self::Session,
        _which: CtrlWhich,
        ctrls: &mut bindings::v4l2_ext_controls,
        ctrl_array: &mut Vec<bindings::v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        for (idx, ctrl) in ctrl_array.iter_mut().enumerate() {
            ctrls.error_idx = idx as u32;
            let err_code = match ctrl.id {
                CID_LENS_FACING => libc::EACCES,
                _ => libc::EINVAL,
            };
            return Err(err_code);
        }
        Ok(())
    }

    fn s_ext_ctrls(
        &mut self,
        _session: &mut Self::Session,
        _which: CtrlWhich,
        ctrls: &mut bindings::v4l2_ext_controls,
        ctrl_array: &mut Vec<bindings::v4l2_ext_control>,
        _user_regions: Vec<Vec<SgEntry>>,
    ) -> IoctlResult<()> {
        for ctrl in ctrl_array {
            ctrls.error_idx = ctrls.count;
            let err_code = match ctrl.id {
                CID_LENS_FACING => libc::EACCES,
                _ => libc::EINVAL,
            };
            return Err(err_code);
        }
        Ok(())
    }

    fn subscribe_event(
        &mut self,
        session: &mut Self::Session,
        event: V4l2EventType,
        flags: SubscribeEventFlags,
    ) -> IoctlResult<()> {
        if !flags.contains(SubscribeEventFlags::SEND_INITIAL) {
            return Err(libc::EINVAL);
        }
        match event {
            V4l2EventType::Ctrl(id) => match id {
                CID_LENS_FACING => {
                    let ctrl_event = bindings::v4l2_event {
                        type_: bindings::V4L2_EVENT_CTRL,
                        id: CID_LENS_FACING,
                        ..Default::default()
                    };
                    let session_id = session.shared.lock().unwrap().id;
                    self.evt_queue
                        .send_event(V4l2Event::Event(SessionEvent::new(session_id, ctrl_event)));
                    Ok(())
                }
                _ => Err(libc::EINVAL),
            },
            _ => Err(libc::EINVAL),
        }
    }

    fn unsubscribe_event(
        &mut self,
        _session: &mut Self::Session,
        event: bindings::v4l2_event_subscription,
    ) -> IoctlResult<()> {
        return if event.type_ == bindings::V4L2_EVENT_CTRL && event.id == CID_LENS_FACING {
            Ok(())
        } else {
            Err(libc::EINVAL)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_write_yuv420_pattern_size() {
        let mut buffer = Vec::new();
        let result = EmulatedCameraSession::write_yuv420_pattern(0, &mut buffer);
        assert!(result.is_ok());
        // Verify output size matches exactly 1.5 bytes per pixel (YUV420p)
        let expected_size = (WIDTH * HEIGHT * 3 / 2) as usize;
        assert_eq!(buffer.len(), expected_size);
    }

    #[test]
    fn test_read_and_write_yuv_frame() {
        let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
        let dummy_input = vec![0x55u8; frame_size];
        let mut source = Cursor::new(dummy_input.clone());
        let mut sink = Vec::new();

        let result = EmulatedCameraSession::read_and_write_yuv_frame(&mut source, &mut sink);
        assert!(result.is_ok());
        assert_eq!(sink, dummy_input);
    }


}
