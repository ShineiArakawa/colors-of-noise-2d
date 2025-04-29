import copy

import cv2
import numpy as np
import pydantic.dataclasses as dataclasses
import pyviewer.docking_viewer as docking_viewer
import research_utilities.apply_color_map as _cm
import research_utilities.plotting as _plotting
import research_utilities.signal as _signal
import research_utilities.torch_util as _torch_util
import torch
from imgui_bundle import imgui, implot

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# load the C++/CUDA module

cuda_enabled = torch.cuda.is_available() and _torch_util.get_extension_loader()._check_command('nvcc')

_module = _signal._get_cpp_module(is_cuda=cuda_enabled, with_omp=True)
print(f'---------------------------------------------------')
print(f'_module.__doc__: {str(_module.__doc__)}')
print(f'---------------------------------------------------')

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Parameters


@dataclasses.dataclass
class Params:
    # autopep8: off
    seed                                : int          = 0
    img_size                            : int          = 512
    std                                 : float        = 1.0
    n_points                            : int          = 1024

    beta                                : float        = 0.0

    scale_input_img_with_minmax         : bool         = False
    scale_psd_with_minmax        : bool         = True
    scale_inv_img_with_minmax           : bool         = False
    scale_inv_img_with_log              : bool         = False
    # autopep8: on

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# utilities


def normalize_0_to_1(x: np.ndarray, based_on_min_max: bool = False) -> np.ndarray:
    if based_on_min_max:
        if x.ndim == 3:
            min = np.min(x, axis=(0, 1), keepdims=True)
            max = np.max(x, axis=(0, 1), keepdims=True)
        elif x.ndim == 2:
            min = np.min(x)
            max = np.max(x)
        else:
            raise ValueError("Input array must be 2D or 3D.")
    else:
        # [-1, 1] -> [0, 1]
        min = -1.0
        max = 1.0

    return np.clip((x - min) / (max - min + 1e-8), 0.0, 1.0)


def to_cuda_device(x: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        return x.cuda()

    return x

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Viewer


class ColorOfNoiseVisualizer(docking_viewer.DockingViewer):
    def __init__(self, name):
        super().__init__(name, with_implot=True)

    def setup_state(self):
        self.state.params = Params()
        self.state.prev_params = None

        self.state.img = None
        self.state.radial_psd = None

        self.state.texture_img = None
        self.state.texture_mag = None
        self.state.texture_inv = None

        self.state.all_img = None

    @property
    def params(self) -> Params:
        return self.state.params

    def _add_title(self, img: np.ndarray, title: str) -> np.ndarray:
        img = (img * 255.0).astype(np.uint8)
        img = _plotting.add_title(
            img,
            title,
            text_color=[255, 255, 255],
            background_color=[0, 0, 0],
            text_size=36,
            is_BGR=False
        )
        img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0.0, 1.0)

        return img

    def compute(self):
        SPATIAL_AXES = (0, 1)
        NUM_RADIAL_BINS = 1024

        # Check if the parameters have changed
        if self.state.prev_params is not None and self.state.prev_params == self.params:
            return self.state.all_img

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # Generate a white noise image
        rand = np.random.RandomState(seed=self.params.seed)
        img = rand.randn(self.params.img_size, self.params.img_size, 3).astype(np.float32) * self.params.std
        self.state.img = img
        assert img.ndim == 3 and img.shape[SPATIAL_AXES[0]] == img.shape[SPATIAL_AXES[1]]

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # FFT
        freq = np.fft.fft2(img, axes=SPATIAL_AXES)
        freq_shifted = np.fft.fftshift(freq, axes=SPATIAL_AXES)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # Calc color of noise
        freq_base_x = np.fft.fftshift(np.fft.fftfreq(img.shape[SPATIAL_AXES[0]], d=(1.0 / img.shape[SPATIAL_AXES[0]])))
        freq_base_y = np.fft.fftshift(np.fft.fftfreq(img.shape[SPATIAL_AXES[1]], d=(1.0 / img.shape[SPATIAL_AXES[1]])))
        freq_base_x, freq_base_y = np.meshgrid(freq_base_x, freq_base_y, indexing='ij')
        freq_base = np.sqrt(freq_base_x ** 2 + freq_base_y ** 2)

        scaling = 1.0 / (np.pow(freq_base, self.params.beta) + 1e-12)

        # Fix the scaling for the DC component
        scaling[scaling.shape[0] // 2, scaling.shape[1] // 2] = 1.0

        freq_shifted = freq_shifted * scaling[:, :, np.newaxis]
        freq_shifted = freq_shifted

        assert freq_shifted.ndim == 3 and freq_shifted.shape[SPATIAL_AXES[0]] == freq_shifted.shape[SPATIAL_AXES[1]]
        assert not np.isnan(freq_shifted).any()
        assert not np.isinf(freq_shifted).any()

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # Calc radial profiles
        spectral_density = np.abs(freq_shifted / (img.shape[SPATIAL_AXES[0]] * img.shape[SPATIAL_AXES[1]]))
        spectral_power_density = spectral_density ** 2

        magnitude = 20.0 * np.log10(spectral_power_density + 1e-10)  # Convert to decibels

        radial_profile: np.ndarray = _module.calc_radial_psd_profile(
            to_cuda_device(torch.from_numpy(spectral_power_density).unsqueeze(0).contiguous()),
            NUM_RADIAL_BINS,
            self.params.n_points,
        ).squeeze(0).cpu().numpy()  # [n_divs, n_points, n_channels]

        assert radial_profile.ndim == 3
        assert radial_profile.shape[0] == NUM_RADIAL_BINS
        assert radial_profile.shape[2] == img.shape[2]
        assert not np.isnan(radial_profile).any()
        assert not np.isinf(radial_profile).any()

        radial_psd: np.ndarray = radial_profile.mean(axis=0)  # [n_points, n_channels]
        self.state.radial_psd = radial_psd.copy()

        assert self.state.radial_psd.ndim == 2
        assert self.state.radial_psd.shape[1] == img.shape[2]
        assert not np.isnan(self.state.radial_psd).any()
        assert not np.isinf(self.state.radial_psd).any()

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # Inverse FFT
        inv_freq_shifted = np.fft.ifftshift(freq_shifted, axes=SPATIAL_AXES)
        inv_img_complex = np.fft.ifft2(inv_freq_shifted, axes=SPATIAL_AXES)
        inv_img = inv_img_complex.real

        assert inv_img.ndim == 3
        assert inv_img.shape[SPATIAL_AXES[0]] == inv_img.shape[SPATIAL_AXES[1]]
        assert inv_img.shape[SPATIAL_AXES[0]] == img.shape[SPATIAL_AXES[0]]
        assert inv_img.shape[SPATIAL_AXES[1]] == img.shape[SPATIAL_AXES[1]]
        assert inv_img.shape[2] == img.shape[2]
        assert not np.isnan(inv_img).any()
        assert not np.isinf(inv_img).any()

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot
        # img
        norm_img = img.astype(np.float32)
        norm_img = normalize_0_to_1(norm_img, based_on_min_max=self.params.scale_input_img_with_minmax)
        self.state.texture_img = norm_img.copy()

        # psd
        # take average over the channels
        psd_img = magnitude.astype(np.float32)
        psd_img = np.mean(psd_img, axis=2)
        psd_img = _cm.apply_color_map(psd_img, 'viridis')  # Returned in BGR format
        psd_img = cv2.cvtColor(psd_img, cv2.COLOR_BGR2RGB)
        psd_img = normalize_0_to_1(psd_img, based_on_min_max=self.params.scale_psd_with_minmax)
        self.state.texture_psd = psd_img.copy()

        # inverse
        inv_img = inv_img.astype(np.float32)
        if self.params.scale_inv_img_with_log:
            inv_img = np.log(np.abs(inv_img) + 1e-10)
        inv_img = normalize_0_to_1(inv_img, based_on_min_max=self.params.scale_inv_img_with_minmax)
        self.state.texture_inv = inv_img.copy()

        # Concat side by side
        all_img = np.concatenate(
            (
                self._add_title(norm_img, 'Input image'),
                self._add_title(psd_img, 'Power spectral density'),
                self._add_title(inv_img, 'Inverse image'),
            ),
            axis=1
        )
        self.state.all_img = all_img.copy()

        self.state.prev_params = copy.deepcopy(self.params)

        return all_img

    @docking_viewer.dockable
    def toolbar(self):
        if imgui.collapsing_header('Noise parameters', flags=imgui.TreeNodeFlags_.default_open):
            self.params.seed = imgui.slider_int('Seed', self.params.seed, 0, 1000)[1]
            self.params.img_size = imgui.slider_int('Image size', self.params.img_size, 128, 1024)[1]
            self.params.std = imgui.slider_float('Std. of Gaussian', self.params.std, 0.0, 10.0)[1]
            self.params.n_points = imgui.slider_int('N points of Radial PSD', self.params.n_points, 512, 1024)[1]

        # ---------------------------------------------------------------------------------------------------
        # Plot radial power spectral density
        if imgui.collapsing_header('Radial power spectral density', flags=imgui.TreeNodeFlags_.default_open):
            if self.state.radial_psd is not None and implot.begin_plot('Radial Power Spectral Density', size=(-1, 512)):
                implot.setup_axes('Frequency [cycles/pixel]', 'Power spectral density [dB]')

                # log-log scale
                implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
                implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)

                # Set axis limits
                implot.setup_axis_limits(implot.ImAxis_.x1, 1.0, len(self.state.radial_psd))
                implot.setup_axis_limits(implot.ImAxis_.y1, 1e-10, 1.0)
                implot.setup_axis_limits_constraints(implot.ImAxis_.x1, 1.0, len(self.state.radial_psd))
                implot.setup_axis_limits_constraints(implot.ImAxis_.y1, 1e-10, 1.0)

                # Fix the view and disable zoom
                implot.setup_axis_zoom_constraints(implot.ImAxis_.x1, len(self.state.radial_psd), len(self.state.radial_psd))
                implot.setup_axis_zoom_constraints(implot.ImAxis_.y1, 1.0, 1.0)

                for i_channel, channel_label in enumerate(['R', 'G', 'B']):
                    radial_psd = np.ascontiguousarray(self.state.radial_psd[:, i_channel])
                    implot.plot_line(f'Radial PSD - {channel_label}', radial_psd)

                implot.end_plot()

        # ---------------------------------------------------------------------------------------------------
        # Color of noise
        if imgui.collapsing_header('Color of noise', flags=imgui.TreeNodeFlags_.default_open):
            self.params.beta = imgui.slider_float('Beta', self.params.beta, -2.0, 2.0)[1]

        # ---------------------------------------------------------------------------------------------------
        # Visualization parameters
        if imgui.collapsing_header('Visualization parameters', flags=imgui.TreeNodeFlags_.default_open):
            imgui.separator_text('Input image')
            self.params.scale_input_img_with_minmax = imgui.checkbox('Scale noise image with min-max', self.params.scale_input_img_with_minmax)[1]

            imgui.separator_text('Power spectral density')
            self.params.scale_psd_with_minmax = imgui.checkbox('Scale PSD with min-max', self.params.scale_psd_with_minmax)[1]

            imgui.separator_text('Inverse image')
            self.params.scale_inv_img_with_minmax = imgui.checkbox('Scale inverse image with min-max', self.params.scale_inv_img_with_minmax)[1]
            self.params.scale_inv_img_with_log = imgui.checkbox('Scale inverse image with log', self.params.scale_inv_img_with_log)[1]

        # ---------------------------------------------------------------------------------------------------
        # Statistics
        if imgui.collapsing_header('Statistics', flags=imgui.TreeNodeFlags_.default_open):
            if self.state.texture_img is not None:
                imgui.separator_text('Input image')
                imgui.text(f'Shape: {self.state.texture_img.shape}')
                imgui.text(f'Min: {np.min(self.state.texture_img):.4f}, Max: {np.max(self.state.texture_img):.4f}, Mean: {np.mean(self.state.texture_img):.4f}, Std: {np.std(self.state.texture_img):.4f}')
            if self.state.texture_psd is not None:
                imgui.separator_text('Power spectral density')
                imgui.text(f'Shape: {self.state.texture_psd.shape}')
                imgui.text(f'Min: {np.min(self.state.texture_psd):.4f}, Max: {np.max(self.state.texture_psd):.4f}, Mean: {np.mean(self.state.texture_psd):.4f}, Std: {np.std(self.state.texture_psd):.4f}')
            if self.state.texture_inv is not None:
                imgui.separator_text('Inverse image')
                imgui.text(f'Shape: {self.state.texture_inv.shape}')
                imgui.text(f'Min: {np.min(self.state.texture_inv):.4f}, Max: {np.max(self.state.texture_inv):.4f}, Mean: {np.mean(self.state.texture_inv):.4f}, Std: {np.std(self.state.texture_inv):.4f}')

        imgui.separator()

        # ---------------------------------------------------------------------------------------------------
        if imgui.button('Reset all params', size=(-1, -1)):
            self.state.params = Params()


if __name__ == '__main__':
    _ = ColorOfNoiseVisualizer('Colors of noise')
    print('Bye!')
