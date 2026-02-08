#!/usr/bin/env python3

import os
import sys
import io
import cv2
import base64
from typing import Callable, Optional, Tuple

import flet as ft
from PIL import Image, ImageOps

import roop.globals
import roop.metadata
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total
from roop.face_reference import (
    get_face_reference,
    set_face_reference,
    clear_face_reference,
)
from roop.predictor import predict_frame, clear_predictor
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None


class RoopApp:
    def __init__(self, start: Callable[[], None], destroy: Callable[[], None]):
        self.start = start
        self.destroy = destroy
        self.page = None
        self.source_label = None
        self.target_label = None
        self.status_text = None
        self.keep_fps_switch = None
        self.keep_frames_switch = None
        self.skip_audio_switch = None
        self.many_faces_switch = None
        self.preview_dialog = None
        self.preview_image = None
        self.preview_slider = None
        self.preview_visible = False
        self.source_picker = None
        self.target_picker = None
        self.output_picker = None

    def build(self, page: ft.Page):
        self.page = page
        page.title = f"{roop.metadata.name} {roop.metadata.version}"
        page.window.width = 600
        page.window.height = 700
        page.window.min_width = 600
        page.window.min_height = 700
        page.theme_mode = ft.ThemeMode.SYSTEM

        self.source_picker = ft.FilePicker()
        self.target_picker = ft.FilePicker()
        self.output_picker = ft.FilePicker()
        self.status_text = ft.Text(text_align=ft.TextAlign.CENTER, width=480)

        self.create_preview_dialog()

        page.add(
            ft.Column(
                [
                    self.create_drop_zones(),
                    self.create_buttons(),
                    self.create_switches(),
                    self.create_action_buttons(),
                    self.status_text,
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=10,
            )
        )

        if roop.globals.source_path:
            self.select_source_path(roop.globals.source_path)
        if roop.globals.target_path:
            self.select_target_path(roop.globals.target_path)

    def create_drop_zones(self) -> ft.Row:
        self.source_label = ft.Container(
            width=200,
            height=175,
            bgcolor=ft.Colors.GREY_300,
            border=ft.border.all(2, ft.Colors.OUTLINE),
            border_radius=8,
            alignment=ft.alignment.Alignment(0.5, 0.5),
            content=ft.Text(
                "Drop source here\nor click to select", text_align=ft.TextAlign.CENTER
            ),
        )

        self.target_label = ft.Container(
            width=200,
            height=175,
            bgcolor=ft.Colors.GREY_300,
            border=ft.border.all(2, ft.Colors.OUTLINE),
            border_radius=8,
            alignment=ft.alignment.Alignment(0.5, 0.5),
            content=ft.Text(
                "Drop target here\nor click to select", text_align=ft.TextAlign.CENTER
            ),
        )

        return ft.Row(
            [self.source_label, self.target_label],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
        )

    def create_buttons(self) -> ft.Row:
        source_button = ft.ElevatedButton(
            "Select a face",
            width=180,
            on_click=self.on_source_click,
        )

        target_button = ft.ElevatedButton(
            "Select a target",
            width=180,
            on_click=self.on_target_click,
        )

        return ft.Row(
            [source_button, target_button],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
        )

    async def on_source_click(self, e):
        files = await self.source_picker.pick_files(
            allowed_extensions=["png", "jpg", "jpeg", "webp"],
            file_type=ft.FilePickerFileType.IMAGE,
        )
        if files and len(files) > 0:
            self.select_source_path(files[0].path)

    async def on_target_click(self, e):
        files = await self.target_picker.pick_files(
            allowed_extensions=[
                "png",
                "jpg",
                "jpeg",
                "webp",
                "mp4",
                "avi",
                "mov",
                "mkv",
            ],
            file_type=ft.FilePickerFileType.ANY,
        )
        if files and len(files) > 0:
            self.select_target_path(files[0].path)

    def create_switches(self) -> ft.Row:
        self.keep_fps_switch = ft.Switch(
            label="Keep target fps",
            value=roop.globals.keep_fps or False,
            on_change=lambda e: setattr(roop.globals, "keep_fps", e.control.value),
        )

        self.keep_frames_switch = ft.Switch(
            label="Keep temporary frames",
            value=roop.globals.keep_frames or False,
            on_change=lambda e: setattr(roop.globals, "keep_frames", e.control.value),
        )

        self.skip_audio_switch = ft.Switch(
            label="Skip target audio",
            value=roop.globals.skip_audio or False,
            on_change=lambda e: setattr(roop.globals, "skip_audio", e.control.value),
        )

        self.many_faces_switch = ft.Switch(
            label="Many faces",
            value=roop.globals.many_faces or False,
            on_change=lambda e: setattr(roop.globals, "many_faces", e.control.value),
        )

        left_column = ft.Column([self.keep_fps_switch, self.keep_frames_switch])
        right_column = ft.Column([self.skip_audio_switch, self.many_faces_switch])

        return ft.Row(
            [left_column, right_column],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=80,
        )

    def create_action_buttons(self) -> ft.Row:
        start_button = ft.ElevatedButton(
            "Start", width=120, on_click=self.on_start_click
        )

        stop_button = ft.ElevatedButton(
            "Destroy",
            width=120,
            on_click=lambda _: self.on_destroy_click(),
        )

        preview_button = ft.ElevatedButton(
            "Preview", width=120, on_click=lambda _: self.toggle_preview()
        )

        return ft.Row(
            [start_button, stop_button, preview_button],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
        )

    async def on_start_click(self, e):
        if is_image(roop.globals.target_path):
            path = await self.output_picker.save_file(
                allowed_extensions=["png", "jpg", "jpeg"],
                file_type=ft.FilePickerFileType.IMAGE,
                initial_directory=RECENT_DIRECTORY_OUTPUT,
                file_name="output.png",
            )
            if path:
                roop.globals.output_path = path
                self.start()
        elif is_video(roop.globals.target_path):
            path = await self.output_picker.save_file(
                allowed_extensions=["mp4", "avi", "mov", "mkv"],
                file_type=ft.FilePickerFileType.ANY,
                initial_directory=RECENT_DIRECTORY_OUTPUT,
                file_name="output.mp4",
            )
            if path:
                roop.globals.output_path = path
                self.start()

    async def on_destroy_click(self):
        if self.page:
            await self.page.window.destroy()
        self.destroy()  # core callback

    def create_preview_dialog(self):
        self.preview_image = ft.Image(
            src="",
            width=PREVIEW_MAX_WIDTH,
            height=PREVIEW_MAX_HEIGHT,
            fit=ft.BoxFit.CONTAIN,
        )
        self.preview_slider = ft.Slider(
            min=0,
            max=0,
            value=0,
            divisions=0,
            on_change=self.on_slider_change,
            visible=False,
        )

        self.preview_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Preview [ ↕ Reference face ]"),
            content=ft.Column([self.preview_image, self.preview_slider], tight=True),
            on_dismiss=self.on_preview_dismiss,
        )

    def on_slider_change(self, e: ft.ControlEvent):
        frame_number = int(e.control.value)
        roop.globals.reference_frame_number = frame_number
        self.update_preview(frame_number)

    def on_preview_dismiss(self, e):
        self.preview_visible = False
        clear_predictor()

    def update_status(self, text: str):
        if self.status_text:
            self.status_text.value = text
            if self.page:
                self.page.update()

    def select_source_path(self, source_path: Optional[str] = None):
        global RECENT_DIRECTORY_SOURCE

        if source_path and is_image(source_path):
            roop.globals.source_path = source_path
            RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
            image = self.render_image_preview(roop.globals.source_path, (200, 200))
            self.update_preview_image(self.source_label, image)
        else:
            roop.globals.source_path = None
            self.update_preview_image(self.source_label, None)
            if self.page:
                self.page.update()

    def select_target_path(self, target_path: Optional[str] = None):
        global RECENT_DIRECTORY_TARGET

        clear_face_reference()
        if target_path and is_image(target_path):
            roop.globals.target_path = target_path
            RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
            image = self.render_image_preview(roop.globals.target_path, (200, 200))
            self.update_preview_image(self.target_label, image)
        elif target_path and is_video(target_path):
            roop.globals.target_path = target_path
            RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
            video_frame = self.render_video_preview(target_path, (200, 200))
            self.update_preview_image(self.target_label, video_frame)
        else:
            roop.globals.target_path = None
            self.update_preview_image(self.target_label, None)
            if self.page:
                self.page.update()

    def update_preview_image(self, container: ft.Container, image: Optional[ft.Image]):
        if image:
            container.content = image
            container.border = None
            container.bgcolor = None
        else:
            container.content = ft.Text(
                "Drop source here\nor click to select", text_align=ft.TextAlign.CENTER
            )
            container.border = ft.border.all(2, ft.Colors.OUTLINE)
            container.bgcolor = ft.Colors.GREY_300
        if self.page:
            self.page.update()

    def render_image_preview(self, image_path: str, size: Tuple[int, int]) -> ft.Image:
        image = Image.open(image_path)
        if size:
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        return self.pil_to_flet_image(image)

    def render_video_preview(
        self, video_path: str, size: Tuple[int, int], frame_number: int = 0
    ) -> Optional[ft.Image]:
        capture = cv2.VideoCapture(video_path)
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if has_frame:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if size:
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            return self.pil_to_flet_image(image)
        capture.release()
        cv2.destroyAllWindows()
        return None

    def pil_to_flet_image(self, pil_image: Image.Image) -> ft.Image:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return ft.Image(
            src=f"data:image/png;base64,{base64_data}",
            width=pil_image.width,
            height=pil_image.height,
        )

    def toggle_preview(self):
        if self.preview_visible:
            self.preview_dialog.open = False
            self.preview_visible = False
            clear_predictor()
            if self.page:
                self.page.update()
        elif roop.globals.source_path and roop.globals.target_path:
            self.init_preview()
            self.update_preview(roop.globals.reference_frame_number)
            self.page.dialog = self.preview_dialog
            self.preview_dialog.open = True
            self.preview_visible = True
            if self.page:
                self.page.update()

    def init_preview(self):
        title = "Preview [ ↕ Reference face ]"
        if is_image(roop.globals.target_path):
            self.preview_slider.visible = False
        elif is_video(roop.globals.target_path):
            video_frame_total = get_video_frame_total(roop.globals.target_path)
            if video_frame_total > 0:
                title += " [ ↔ Frame number ]"
                self.preview_slider.max = video_frame_total
                self.preview_slider.value = roop.globals.reference_frame_number
                self.preview_slider.visible = True
            else:
                self.preview_slider.visible = False
        self.preview_dialog.title.value = title

    def update_preview(self, frame_number: int = 0):
        if roop.globals.source_path and roop.globals.target_path:
            temp_frame = get_video_frame(roop.globals.target_path, frame_number)
            if predict_frame(temp_frame):
                sys.exit()
            source_face = get_one_face(cv2.imread(roop.globals.source_path))
            if not get_face_reference():
                reference_frame = get_video_frame(
                    roop.globals.target_path, roop.globals.reference_frame_number
                )
                reference_face = get_one_face(
                    reference_frame, roop.globals.reference_face_position
                )
                set_face_reference(reference_face)
            else:
                reference_face = get_face_reference()
            for frame_processor in get_frame_processors_modules(
                roop.globals.frame_processors
            ):
                temp_frame = frame_processor.process_frame(
                    source_face, reference_face, temp_frame
                )
            image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
            image = ImageOps.contain(
                image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.Resampling.LANCZOS
            )
            self.preview_image.src = self.pil_image_to_base64(image)
            self.preview_image.width = image.width
            self.preview_image.height = image.height
            if self.page:
                self.page.update()

    def pil_image_to_base64(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

    def update_face_reference(self, steps: int):
        clear_face_reference()
        reference_frame_number = int(self.preview_slider.value)
        roop.globals.reference_face_position += steps
        roop.globals.reference_frame_number = reference_frame_number
        self.update_preview(reference_frame_number)

    def update_frame(self, steps: int):
        frame_number = self.preview_slider.value + steps
        self.preview_slider.value = max(0, min(frame_number, self.preview_slider.max))
        self.update_preview(int(self.preview_slider.value))


_app_instance = None


def mainloop(start: Callable[[], None], destroy: Callable[[], None]):
    global _app_instance
    _app_instance = RoopApp(start, destroy)
    ft.run(_app_instance.build)


def update_status(text: str):
    if _app_instance:
        _app_instance.update_status(text)
