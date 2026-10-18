import asyncio, sys, json
from playwright.async_api import async_playwright
URL = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "x"
async def run(p, name, vp, mobile):
    b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    ctx = await b.new_context(viewport=vp, is_mobile=mobile, has_touch=mobile, device_scale_factor=2 if mobile else 1)
    pg = await ctx.new_page(); errs = []
    pg.on("console", lambda m: errs.append(("console." + m.type, m.text)) if m.type in ("error", "warning") and "GL Driver Message" not in m.text else None)
    pg.on("pageerror", lambda e: errs.append(("pageerror", str(e))))
    await pg.goto(URL, wait_until="domcontentloaded", timeout=60000); await pg.wait_for_timeout(8000)
    r = {}
    r["menu_open_initial"] = await pg.evaluate("!document.getElementById('controls').classList.contains('hidden')")
    await pg.screenshot(path=f"/workspace/viewer/{tag}_{name}_closed.png")
    await pg.click("#menubtn"); await pg.wait_for_timeout(500)
    r["menu_open_after_click"] = await pg.evaluate("!document.getElementById('controls').classList.contains('hidden')")
    r["menu_box"] = await pg.evaluate("(()=>{const b=document.getElementById('controls').getBoundingClientRect();return [Math.round(b.left),Math.round(b.top),Math.round(b.width),Math.round(b.height)]})()")
    if not r["menu_open_after_click"]:
        await pg.screenshot(path=f"/workspace/viewer/{tag}_{name}_toggled_closed.png"); await pg.click("#menubtn"); await pg.wait_for_timeout(500)
    r["menu_box"] = await pg.evaluate("(()=>{const b=document.getElementById('controls').getBoundingClientRect();return [Math.round(b.left),Math.round(b.top),Math.round(b.width),Math.round(b.height)]})()")
    await pg.screenshot(path=f"/workspace/viewer/{tag}_{name}_open.png")
    # log: turn on, then off; wait for state broadcasts with new log lines; must stay off; reload must keep it off
    await pg.check("#showlog"); await pg.wait_for_timeout(300); await pg.uncheck("#showlog")
    await pg.evaluate("fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'stop',who:'uitest'})})")
    await pg.wait_for_timeout(4000)
    r["log_off_after_msgs"] = await pg.evaluate("!document.getElementById('showlog').checked && !document.getElementById('log').classList.contains('show')")
    r["unread_badge"] = await pg.evaluate("document.getElementById('menubtn').classList.contains('unread')")
    await pg.reload(wait_until="domcontentloaded"); await pg.wait_for_timeout(6000)
    r["log_off_after_reload"] = await pg.evaluate("!document.getElementById('showlog').checked && !document.getElementById('log').classList.contains('show')")
    r["menu_remembered_open"] = await pg.evaluate("!document.getElementById('controls').classList.contains('hidden')")
    r["errors"] = errs
    await b.close(); return r
async def main():
    async with async_playwright() as p:
        out = {"phone_412x915": await run(p, "phone", {"width": 412, "height": 915}, True),
               "desktop_1440x900": await run(p, "desktop", {"width": 1440, "height": 900}, False)}
    print(json.dumps(out)); open("/workspace/viewer/result_%s.json" % tag, "w").write(json.dumps(out, indent=1))
asyncio.run(main())
