import matplotlib.pyplot as plt
import moderngl
import numpy as np
from PIL import Image

from icecream import ic


def rasterize_sdf(sdf, dim):
    ctx = moderngl.create_context(standalone=True, backend="egl")

    vert = open("shader/fullscreen.vs").read()
    frag = open("shader/sdf.fs").read()

    prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

    color_attachments = [ctx.texture((dim, dim), 4, dtype="f4")]
    fbo = ctx.framebuffer(color_attachments=color_attachments)

    vbo_vert_screen_space = ctx.buffer(
        np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]).astype("f4")
    )
    vbo_uv_screen_space = ctx.buffer(
        np.array([[0.0, 1.0], [2.0, 1.0], [0.0, -1.0]]).astype("f4")
    )
    ibo_screen_space = ctx.buffer(np.array([0, 1, 2]).astype("i4"))

    sdf_img = Image.fromarray(np.array(sdf))
    texture = ctx.texture(sdf_img.size, 1, sdf_img.tobytes(), dtype="f4")
    texture.build_mipmaps()

    vao = ctx.vertex_array(
        prog,
        [(vbo_vert_screen_space, "2f", "in_pos"), (vbo_uv_screen_space, "2f", "in_uv")],
        ibo_screen_space,
    )

    fbo.use()
    fbo.clear()
    texture.use()
    vao.render()

    render_data = fbo.read(components=4, attachment=0, dtype="f4")
    render_img = np.frombuffer(render_data, dtype="f4").reshape(dim, dim, 4)
    render_img = Image.fromarray(np.int8(render_img * 255), mode="RGBA")
    return render_img


class PatchworkRasterizer:
    def __init__(self, dim):
        ctx = moderngl.create_context(standalone=True, backend="egl")
        # Cannot handle more due to uniform buffer restriction
        self.MAX_LINES = 256

        vert = open("shader/fullscreen.vs").read()
        frag = open("shader/patchwork.fs").read()

        # https://www.shadertoy.com/view/3ltSW2
        self.prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        self.dim = dim
        color_attachments = [ctx.texture((dim, dim), 4, dtype="f4")]
        self.fbo = ctx.framebuffer(color_attachments=color_attachments)

        vbo_vert_screen_space = ctx.buffer(
            np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]).astype("f4")
        )
        vbo_uv_screen_space = ctx.buffer(
            np.array([[0.0, 1.0], [2.0, 1.0], [0.0, -1.0]]).astype("f4")
        )
        ibo_screen_space = ctx.buffer(np.array([0, 1, 2]).astype("i4"))

        self.vao = ctx.vertex_array(
            self.prog,
            [
                (vbo_vert_screen_space, "2f", "in_pos"),
                (vbo_uv_screen_space, "2f", "in_uv"),
            ],
            ibo_screen_space,
        )

    def rasterize(
        self,
        coeffs,
        masks,
        beta,
        line_scale,
        use_softmax=True,
        draw_candidate=True,
        pad_eps=False,
    ):
        num_lines = len(coeffs)
        if num_lines > self.MAX_LINES:
            num_lines = min(self.MAX_LINES, num_lines)
            print(
                "Coeffs size exceeds uniform buffer limits. Exceeding lines will be ignored."
            )

        coeffs_uniform = np.zeros((self.MAX_LINES, 4))
        coeffs_uniform[:num_lines] = coeffs[:num_lines]

        mask_uniform = np.zeros((self.MAX_LINES,))
        mask_uniform[:num_lines] = masks[:num_lines]

        # Uniforms
        self.prog["iNum"] = num_lines
        self.prog["fCoeffs"] = coeffs_uniform.astype("f4")
        self.prog["iEnable"] = mask_uniform.astype("i4")
        self.prog["fBeta"] = beta
        self.prog["iSoftmax"] = use_softmax
        self.prog["iCandidate"] = draw_candidate
        self.prog["iPadEps"] = pad_eps
        self.prog["fLineScale"] = line_scale

        self.fbo.use()
        self.fbo.clear()
        self.vao.render()

        render_data = self.fbo.read(components=3, attachment=0, dtype="f4")
        render_img = np.frombuffer(render_data, dtype="f4").reshape(
            self.dim, self.dim, 3
        )
        return render_img
