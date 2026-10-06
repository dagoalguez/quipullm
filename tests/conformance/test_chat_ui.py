"""Saved chats in the /chat page (IndexedDB), checked in headless Chromium against a simulated server.
Run: python tests/conformance/test_chat_ui.py   (needs Playwright; not part of run_all.py)"""
import asyncio, json
from playwright.async_api import async_playwright
import os, tempfile
AQUI=os.path.dirname(os.path.abspath(__file__))
RAIZ=os.path.dirname(os.path.dirname(AQUI))
TMP=tempfile.mkdtemp()
HTML=open(os.path.join(RAIZ,"web","chat.html"),encoding="utf-8").read()
ST={"requires_key":False,"engine":{"connected":True,"loaded_model":"m"},"current_job":None,"queued":0,"panel_allowed":True,"models":[{"id":"m","supported":True,"embedding":False,"ctx":8192}],"chat_models":[]}
sent=[]; res=[]; claves=[]
def chk(n,c,x=""): res.append(bool(c)); print(("OK   " if c else "FALLA"),n,("" if c else x))
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); ctx=await b.new_context(accept_downloads=True,viewport={"width":1200,"height":800}); pg=await ctx.new_page(); errs=[]; pg.on("pageerror",lambda e:errs.append(str(e)))
        dlg=[]
        async def ondlg(d):
            dlg.append((d.type,d.message)); await d.accept(d.default_value if d.type=="prompt" else None) if d.type!="prompt" else await d.accept(NOMBRE[0])
        NOMBRE=["Mi chat renombrado"]
        pg.on("dialog",lambda d: asyncio.ensure_future(ondlg(d)))
        async def h(r):
            u=r.request.url
            if u.endswith("/chat"): return await r.fulfill(body=HTML,content_type="text/html")
            if "/api/status" in u: return await r.fulfill(body=json.dumps(ST),content_type="application/json")
            if "/v1/chat" in u:
                msgs=json.loads(r.request.post_data)["messages"]; sent.append(len(msgs)); claves.append(sorted({k for m in msgs for k in m}))
                body="data: "+json.dumps({"model":"modelo-x","choices":[{"delta":{"content":"respuesta **%d**"%len(msgs)}}]})+"\n\ndata: "+json.dumps({"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":5}})+"\n\ndata: [DONE]\n\n"
                return await r.fulfill(body=body,content_type="text/event-stream")
            await r.fulfill(status=404,body="")
        await pg.route("http://t.test/**",h); await pg.goto("http://t.test/chat"); await pg.wait_for_timeout(500)
        chk("lateral abierto en pantalla ancha", await pg.locator("#lateral").is_visible())
        chk("lista vacía muestra aviso", await pg.locator(".sinchats").count()==1)
        async def decir(x):
            await pg.fill("#texto",x); await pg.click("#enviar"); await pg.wait_for_timeout(500)
        await decir("primera pregunta"); await decir("segunda pregunta")
        chk("1 chat guardado en la lista", await pg.locator(".item").count()==1)
        chk("título = primer mensaje", (await pg.locator(".item .tit").first.inner_text())=="primera pregunta")
        await pg.reload(); await pg.wait_for_timeout(600)
        chk("tras F5 el chat sigue en la lista", await pg.locator(".item").count()==1)
        chk("tras F5 la conversación está vacía (hasta abrir)", await pg.locator(".msg").count()==0)
        await pg.click(".item .tit"); await pg.wait_for_timeout(300)
        chk("abrir restaura 4 burbujas", await pg.locator(".msg").count()==4)
        chk("Markdown restaurado (negrita)", await pg.locator(".msg.assistant strong").count()==2)
        meta=await pg.locator(".msg.assistant .meta").first.inner_text()
        chk("cada respuesta muestra modelo, tokens, primer token", meta.startswith("modelo-x · 5 tokens · first token in") , meta)
        chk("la petición al servidor solo lleva role y content", all(k==["content","role"] for k in claves), claves)
        chk("solo la última respuesta tiene Regenerar", await pg.locator(".regen").count()==1)
        sent.clear(); await decir("tercera pregunta")
        chk("al continuar se envía el historial restaurado (5 msgs)", sent==[5], sent)
        chk("sigue siendo 1 solo chat", await pg.locator(".item").count()==1)
        await pg.click("#nuevo"); await pg.wait_for_timeout(200)
        chk("nuevo chat limpia pantalla", await pg.locator(".msg").count()==0)
        await decir("otro tema")
        chk("2 chats en la lista", await pg.locator(".item").count()==2)
        chk("el más reciente va primero", (await pg.locator(".item .tit").first.inner_text())=="otro tema")
        # renombrar
        await pg.locator(".item").first.hover(); await pg.locator(".item .ic").first.click(); await pg.wait_for_timeout(300)
        chk("renombrar", (await pg.locator(".item .tit").first.inner_text())=="Mi chat renombrado")
        await pg.reload(); await pg.wait_for_timeout(600)
        chk("nombre persiste tras F5", (await pg.locator(".item .tit").first.inner_text())=="Mi chat renombrado")
        # exportar
        async with pg.expect_download() as di: await pg.click("#exportar")
        d=await di.value; path=await d.path(); data=json.load(open(path,encoding="utf-8"))
        chk("export JSON válido con 2 chats", data["app"]=="quipullm" and len(data["chats"])==2)
        open(os.path.join(TMP,"export.json"),"w",encoding="utf-8").write(json.dumps(data))
        # borrar uno
        await pg.locator(".item").nth(1).hover(); await pg.locator(".item").nth(1).locator(".ic").nth(1).click(); await pg.wait_for_timeout(300)
        chk("borrar uno", await pg.locator(".item").count()==1)
        # borrar todo e importar
        await pg.click("#borrarTodo"); await pg.wait_for_timeout(300)
        chk("borrar todos", await pg.locator(".item").count()==0)
        await pg.set_input_files("#archivo",os.path.join(TMP,"export.json")); await pg.wait_for_timeout(500)
        chk("importar restaura 2 chats", await pg.locator(".item").count()==2 and any("Imported 2" in m for _,m in dlg), dlg[-1:])
        open(TMP+"/malo.json","w").write(json.dumps({"app":"x","chats":[]}))
        await pg.set_input_files("#archivo",TMP+"/malo.json"); await pg.wait_for_timeout(400)
        chk("archivo inválido rechazado", any("not a valid" in m for _,m in dlg))
        open(TMP+"/malo2.json","w").write(json.dumps({"app":"quipullm","chats":[{"id":"z","mensajes":[{"role":"system","content":"x"}]},{"id":"y","mensajes":[{"role":"user","content":"<img src=x onerror=alert(1)>"}]}]}))
        await pg.set_input_files("#archivo",TMP+"/malo2.json"); await pg.wait_for_timeout(400)
        chk("importa solo los válidos (rol system descartado)", await pg.locator(".item").count()==3)
        await pg.locator(".item .tit",has_text="img").click(); await pg.wait_for_timeout(200)
        chk("texto importado no se interpreta como HTML", await pg.locator("#mensajes img").count()==0)
        # retención
        viejo=json.dumps({"app":"quipullm","chats":[{"id":"viejo","titulo":"antiguo","actualizado":1,"creado":1,"mensajes":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}]})
        open(TMP+"/viejo.json","w").write(viejo)
        await pg.set_input_files("#archivo",TMP+"/viejo.json"); await pg.wait_for_timeout(400)
        n0=await pg.locator(".item").count()
        chk("chat antiguo importado (retención nunca)", any("antiguo"==x for x in await pg.locator(".item .tit").all_inner_texts()))
        await pg.select_option("#retencion","30"); await pg.wait_for_timeout(400)
        chk("retención 30 días borra el antiguo y conserva los recientes", (await pg.locator(".item").count())==n0-1 and not any("antiguo"==x for x in await pg.locator(".item .tit").all_inner_texts()))
        await pg.reload(); await pg.wait_for_timeout(500)
        chk("retención persiste tras F5", await pg.input_value("#retencion")=="30")
        # info por respuesta en importación
        open(os.path.join(TMP,"info.json"),"w").write(json.dumps({"app":"quipullm","chats":[{"id":"inf","titulo":"conInfo","mensajes":[{"role":"user","content":"q"},{"role":"assistant","content":"a","info":{"modelo":"mm","n":3,"ttft":"x","tps":2.5,"extra":"<b>"}}]}]}))
        await pg.set_input_files("#archivo",os.path.join(TMP,"info.json")); await pg.wait_for_timeout(400)
        await pg.locator(".item .tit",has_text="conInfo").click(); await pg.wait_for_timeout(200)
        chk("importación valida info (ttft inválido se descarta)", (await pg.locator(".msg.assistant .meta").first.inner_text()).startswith("mm · 3 tokens · 2.5 tok/s"), await pg.locator(".msg.assistant .meta").first.inner_text())
        # tope
        await pg.evaluate("""async()=>{ for(let i=0;i<305;i++){ const c={id:'c'+i,titulo:'t'+i,creado:1000+i,actualizado:Date.now()-1e6+i,mensajes:[{role:'user',content:'a'},{role:'assistant',content:'b'}]}; lista.push(c); await BD.poner(c);} }""")
        await pg.click("#nuevo"); await decir("dispara guardado")
        chk("tope de 300 chats", await pg.evaluate("lista.length")==300 and len(await pg.evaluate("BD.todos()"))==300)
        # cambiar de chat mientras genera no corrompe
        # móvil
        pm=await ctx.new_page(); await pm.set_viewport_size({"width":390,"height":800}); await pm.route("http://t.test/**",h); await pm.goto("http://t.test/chat")
        await pm.evaluate("localStorage.removeItem('quipullm_lateral')"); await pm.reload(); await pm.wait_for_timeout(500)
        chk("en móvil el lateral empieza cerrado", not await pm.locator("#lateral").is_visible())
        await pm.click("#lat"); chk("botón ☰ lo abre", await pm.locator("#lateral").is_visible())
        chk("sin errores de JS", not errs, errs)
        await b.close()
asyncio.run(main())

async def panel_fijo():
    from playwright.async_api import async_playwright as ap
    PANEL=open(os.path.join(RAIZ,"web","panel.html"),encoding="utf-8").read()
    async with ap() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={"width":1100,"height":500})
        async def h(r):
            if r.request.url.endswith("/"): return await r.fulfill(body=PANEL,content_type="text/html")
            await r.fulfill(status=404,body="{}",content_type="application/json")
        await pg.route("http://t.test/**",h); await pg.goto("http://t.test/"); await pg.wait_for_timeout(500)
        await pg.evaluate("document.getElementById('bannerCarga').style.display='block'; document.body.style.minHeight='3000px'")
        await pg.mouse.wheel(0,1500); await pg.wait_for_timeout(300)
        top=await pg.evaluate("[document.querySelector('header').getBoundingClientRect().top, document.getElementById('bannerCarga').getBoundingClientRect().top]")
        chk("panel: encabezado y banner de carga siguen arriba al desplazar", await pg.evaluate("scrollY")>1000 and top[0]==0 and 0<top[1]<120, top)
        await b.close()
asyncio.run(panel_fijo())
print("TOTAL", sum(res), "OK,", len(res)-sum(res), "fallas")
import sys; sys.exit(1 if not all(res) else 0)
