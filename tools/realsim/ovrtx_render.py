"""ovrtx (Omniverse RTX, RealTimePathTracing) driver for the casita stage of tools/realsim/usd_scene.py.

RtxSim(usda, settings) opens the stage through ovstage, moves the cameras with ovstage matrix writes (omni:xform,
USD row-vector convention) and steps the renderer; frames come back as numpy (HdrColor linear float, LdrColor uint8).
Settings (dict USD-attr -> (type, value)) are authored on every RenderProduct as an 'over' layer, so one process = one
configuration (a bench sweeps configurations in separate processes).
Shader cache: XDG_CACHE_HOME (we keep it in /dev/shm/ovcache on i777: ~0.9 GB, first build ~11 min, warm start ~10 s).
"""
import os, time
import numpy as np
import ovrtx, ovstage


def _fmt(typ, v):
    if typ == "bool": return "1" if v else "0"
    if typ == "token" or typ == "string": return '"%s"' % v
    return str(v)


class RtxSim:
    def __init__(self, usda, products, settings=None, log=None):
        self.products = list(products); self.ordinal = 1
        ov = "".join('    over "%s"\n    {\n%s    }\n' % (p, "".join("        %s %s = %s\n" % (t, k, _fmt(t, v)) for k, (t, v) in (settings or {}).items()))
                     for p in self.products)
        scene = '#usda 1.0\n(\n    subLayers = [@%s@]\n)\nover "Render"\n{\n%s}\n' % (os.path.abspath(usda), ov)
        self.r = ovrtx.Renderer(config=ovrtx.RendererConfig(log_file_path=log))
        self.stage = ovstage.Stage("kevin.realsim"); self.r.attach_ovstage(self.stage)
        ovstage.population.open_usd_from_string(self.stage, scene, ordinal=self.ordinal)
        self.stage.advance_write_floor(self.ordinal, ovstage.Scope.ALL).wait()
        self.paths = ovstage.PathDictionary(self.stage)
        self._attr = self.paths.intern_token("omni:xform"); self._q = {}

    def set_cams(self, mats):
        """mats: {camera prim path: 4x4 local-to-world (row-vector convention)}; one ordinal for all."""
        self.ordinal += 1
        for path, M in mats.items():
            if path not in self._q:
                pl = self.paths.create_path_list_from_strings([path]); self._q[path] = (pl, self.stage.query_from_path_list(pl))
            X = np.ascontiguousarray(np.asarray(M, np.float64).reshape(1, 4, 4))
            t = ovstage.make_dltensor(X, dtype=ovstage.numpy_to_dldatatype(X.dtype, lanes=16), shape=[1], ndim=1)
            self.stage.write_attribute(self._q[path][1], self._attr, ordinal=self.ordinal, tensors=t, is_array=False,
                                       semantic=ovstage.AttributeSemantic.MATRIX).wait()
        self.stage.advance_write_floor(self.ordinal, ovstage.Scope.ALL).wait()

    def step(self, products=None, dt=1.0 / 30, read=("HdrColor",)):
        products = products or self.products; out = {}
        res = self.r.step(render_products={"/Render/" + p for p in products}, delta_time=dt, ordinal=self.ordinal)
        for p in products:
            for fr in res["/Render/" + p].frames:
                for v in read:
                    m = fr.render_vars["/Render/%s/%s" % (p, v)].map(device=ovrtx.Device.CPU)
                    out[(p, v)] = np.from_dlpack(m).copy(); m.unmap(); del m
                del fr
        del res
        return out

    def close(self):
        for pl, q in self._q.values():
            try: q.__exit__(None, None, None)
            except Exception: pass
        try: self.r.detach_ovstage(); self.stage.destroy(); self.r.destroy()
        except Exception: pass


def hdr_to_rgb(x):
    x = np.asarray(x, np.float32)
    return x[..., :3]
