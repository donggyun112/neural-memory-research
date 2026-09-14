import numpy as np
from PIL import Image

from image_to_photoreceptors import (
    color_proxy_activation,
    contrast_map,
    r1_r6_activation,
    r1_r6_retinotopic_activation,
)


def _solid_image(rgb, size=(8, 8)):
    return Image.new("RGB", size, rgb)


def _edge_image(size=200):
    """Left half black, right half white -- a real edge for contrast to find."""
    image = Image.new("L", (size, size), 0)
    for x in range(size // 2, size):
        for y in range(size):
            image.putpixel((x, y), 255)
    return image.convert("RGB")


def test_contrast_map_is_near_zero_for_a_flat_image():
    contrast = contrast_map(_solid_image((128, 128, 128)))
    assert contrast.max() < 1e-6


def test_contrast_map_is_higher_near_an_edge_than_in_a_flat_region():
    contrast = contrast_map(_edge_image(size=200), blur_radius=10)
    near_edge = contrast[100, 95:105].mean()
    far_from_edge = contrast[100, 5:15].mean()
    assert near_edge > far_from_edge


def test_r1_r6_activation_has_the_requested_length():
    image = _solid_image((128, 128, 128))
    activation = r1_r6_activation(image, n_neurons=50)
    assert activation.shape == (50,)
    assert (activation >= 0).all() and (activation <= 1).all()


def test_r1_r6_activation_is_near_zero_for_a_flat_image():
    activation = r1_r6_activation(_solid_image((200, 200, 200)), n_neurons=20)
    assert activation.max() < 0.05


def test_color_proxy_activation_picks_the_right_channel():
    image = _solid_image((200, 10, 10))  # pure-ish red
    yellow_group = color_proxy_activation(image, "R7y", n_neurons=5)
    pale_group = color_proxy_activation(image, "R7p", n_neurons=5)
    assert yellow_group[0] > pale_group[0]


def test_r1_r6_retinotopic_activation_finds_more_contrast_near_the_edge():
    image = _edge_image(size=200)
    body_ids = np.array([1, 2])
    retino_body_ids = np.array([1, 2])
    # neuron 1 sits right at the edge (u=0.5), neuron 2 sits far from it (u=0.05)
    retino_uv = np.array([[0.5, 0.5], [0.05, 0.5]])

    activation = r1_r6_retinotopic_activation(image, body_ids, retino_body_ids, retino_uv)
    assert activation[0] > activation[1]


def test_r1_r6_retinotopic_activation_falls_back_to_center_for_unknown_body():
    image = Image.new("RGB", (4, 4), (128, 128, 128))
    activation = r1_r6_retinotopic_activation(image, np.array([999]), np.array([1]), np.array([[0.0, 0.0]]))
    assert activation.shape == (1,)
