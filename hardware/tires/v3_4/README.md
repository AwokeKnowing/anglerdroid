# Soft tire v3.4 (TPU 95A)

- v3.2 was the version printed/ordered. v3.4 is for future prints.
- v3.4 changes: spoke root fillets (1.75 mm) follow the curved inner ring across full width (fixes V-notches at spoke/ring junction seen in v3.2); 8 countersunk screw tabs on BOTH faces (32 screws total); ~2 mm wider, ~180 g, ~$26/tire; flat tread.
- Outer-face tab holes are ASSUMED to match the inner-face hole pattern — verify on the wheel before printing.
- Outer tabs need print supports.
- Use `softtire_v3_4_tire_95a.stl`. The STEP export is not valid yet and is omitted.
- Regenerate: `python gen_softtire_v3_4.py` then `python build_v3_4_mesh.py`.
- Dependencies: gen_softtire_v3.py, gen_softtire_v3_2.py and softtire_v3_2_tire_95a.stl (the printed v3.2 base mesh) are included in deps/ — run from a dir containing them.
