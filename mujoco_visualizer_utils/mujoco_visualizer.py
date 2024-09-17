import json
from pathlib import Path

import cv2
import mujoco
import tqdm


class MuJoCoVisualizer:

    def __init__(self, render_fps: int = 30, render_size: (int, int) = (1280, 1280)):
        self.mjc_model = None
        self.mjc_data = None
        self.renderer = None
        self.data = {}
        self.render_fps = render_fps
        self.render_size = render_size
        self.camera = "fixed"

    def set_camera(self, camera_name):
        self.camera = camera_name

    def set_xml_path(self, xml_path: Path):
        self.mjc_model = self._load_model_from_xml(xml_path)
        self.mjc_data = mujoco.MjData(self.mjc_model)
        self.renderer = mujoco.Renderer(self.mjc_model, self.render_size[0], self.render_size[1])

        mujoco.mj_resetData(self.mjc_model, self.mjc_data)

    def _load_model_from_xml(self, xml_path: Path) -> mujoco.MjModel:
        model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        return model

    def load_data(self, data_path: Path):
        with data_path.open("r") as fp:
            self.data = json.load(fp)

    def visualize(self, save_video_path, dt, data=None):
        frames = []
        num_steps_per_frame = int(1 / self.render_fps / dt)
        for i, data_step in tqdm.tqdm(enumerate(self.data)):
            # if True:
            if i % num_steps_per_frame == 0:
                frame = self.take_snap_shot(data_step['time'],
                                            data_step['pos'])

                frames.append(frame)
                # cv2.imwrite(Path(save_video_path, f"{i}.png").as_posix(), frame)

        self.save_video(save_video_path, frames)

    def visualize_from_ext_data(self, xml_path, data_path, dt, video_path):
        self.mjc_model = self._load_model_from_xml(xml_path)
        self.mjc_data = mujoco.MjData(self.mjc_model)
        self.renderer = mujoco.Renderer(self.mjc_model, self.render_size[0], self.render_size[1])
        self.load_data(data_path)

        self.visualize(video_path, dt)

    def render_frame(self):
        self.renderer.update_scene(self.mjc_data, self.camera)
        frame = self.renderer.render()
        return frame

    def save_video(self, save_path: Path, frames: list):
        frame_size = (self.renderer.width, self.renderer.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path.as_posix(), fourcc, self.render_fps, frame_size)

        for i, frame in enumerate(frames):
            im = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(im)

        video_writer.release()

    def take_snap_shot(self, t, pos, camera_view=None):
        self.mjc_data.time = t
        self.mjc_data.qpos = pos

        mujoco.mj_forward(self.mjc_model, self.mjc_data)
        self.renderer.update_scene(self.mjc_data, camera_view if camera_view else self.camera)
        frame = self.renderer.render().copy()

        return frame
