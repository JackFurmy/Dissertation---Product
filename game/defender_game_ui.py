from __future__ import annotations
import os, sys, textwrap, tempfile
from typing import Dict, List, Tuple

import pygame
import pandas as pd
import requests                          

from defender_game import (                   
    SCENARIOS, get_csv, _labelcol, delta_to_points
)
from ml_integration_defender import get_model_manager  

from user_session  import update_user_score, get_user_score, add_defender_score
from firebase_init import get_firestore_db


API_BASE = os.getenv(
    "CYBER_ML_API_BASE",
    "URL"  
).rstrip("/")

def backend_ingest(scenario_key: str, chunk_url: str) -> Dict[str, float]:
    r = requests.post(f"{API_BASE}/defender/update",
                      json={"scenario_key": scenario_key,
                            "chunk_csv_url": chunk_url},
                      timeout=900)
    r.raise_for_status()
    return r.json()


def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel_path)

if getattr(sys, "frozen", False):
    os.chdir(sys._MEIPASS)                     

WIDTH, HEIGHT = 1200, 800
pygame.init()

FONT_HUGE  = pygame.font.SysFont(None, 48)
FONT_BIG   = pygame.font.SysFont(None, 40)
FONT_MED   = pygame.font.SysFont(None, 30)
FONT_HELP  = pygame.font.SysFont(None, 26)
FONT_SMALL = pygame.font.SysFont(None, 22)
FONT_UID   = pygame.font.SysFont(None, 24)

COLOR_BG,  COLOR_TXT  = (245, 255, 246), (0, 0, 0)
COLOR_BTN, COLOR_HOV  = (132, 145, 163), (150, 163, 180)
COLOR_CARD            = (233, 239, 232)

CHIP_COLOR   = (240, 248, 235)
CHIP_OUTLINE = (200, 208, 195)
CHIP_H, CHIP_R = 36, 18
CHIP_MARGIN  = 20
LOGOUT_W, LOGOUT_H = 100, 36

BTN_H, GAP                 = 64, 12
LIST_W, SCROLL_W, SCROLL_GAP = 480, 16, 6
PREVIEW_N                  = 300
COLOR_SELECTED_FILL        = (207, 224, 255)

def _shade(rgb, d): return tuple(max(0, min(255, c + d)) for c in rgb)

def draw_panel(surf, rect, *, fill=COLOR_CARD):
    pygame.draw.rect(surf, fill, rect, border_radius=10)
    pygame.draw.line(surf, _shade(fill, 45),
                     (rect.left+1, rect.top+1),
                     (rect.right-2, rect.top+1), 2)
    pygame.draw.line(surf, _shade(fill, 45),
                     (rect.left+1, rect.top+1),
                     (rect.left+1, rect.bottom-2), 2)
    pygame.draw.line(surf, _shade(fill, -45),
                     (rect.left+1, rect.bottom-2),
                     (rect.right-2, rect.bottom-2), 2)
    pygame.draw.line(surf, _shade(fill, -45),
                     (rect.right-2, rect.top+1),
                     (rect.right-2, rect.bottom-2), 2)

def draw_button(surf, rect, txt, hover, font=FONT_MED):
    fill = COLOR_HOV if hover else COLOR_BTN
    pygame.draw.rect(surf, fill, rect, border_radius=8)
    t = font.render(txt, True, (255, 255, 255))
    surf.blit(t, (rect.centerx - t.get_width()//2,
                  rect.centery - t.get_height()//2))

def draw_uid_chip(surface, uid_text, logout_rect=None):
    uid_surf = FONT_UID.render(f"UID: {uid_text}", True, COLOR_TXT)
    chip_w = (uid_surf.get_width()+24 if logout_rect is None
              else uid_surf.get_width()+logout_rect.width+36)
    chip_x = WIDTH-chip_w-CHIP_MARGIN
    chip_y = CHIP_MARGIN
    chip_rect = pygame.Rect(chip_x, chip_y, chip_w, CHIP_H)

    pygame.draw.rect(surface, CHIP_COLOR, chip_rect, border_radius=CHIP_R)
    pygame.draw.rect(surface, CHIP_OUTLINE, chip_rect, 1, border_radius=CHIP_R)
    surface.blit(uid_surf, (chip_x+12,
                            chip_y+(CHIP_H-uid_surf.get_height())//2))

    if logout_rect:
        logout_rect.y = chip_y
        logout_rect.x = chip_rect.right-logout_rect.width-12
        pygame.draw.rect(surface, CHIP_COLOR, logout_rect, border_radius=CHIP_R)
        pygame.draw.rect(surface, CHIP_OUTLINE, logout_rect, 1, border_radius=CHIP_R)
        lo_txt = FONT_UID.render("Log Out", True, COLOR_TXT)
        surface.blit(lo_txt, (logout_rect.centerx-lo_txt.get_width()//2,
                              logout_rect.centery-lo_txt.get_height()//2))

def wrap_text(txt: str, font, max_w: int) -> List[str]:
    out: List[str] = []
    for para in txt.split("\n"):
        if not para:
            out.append(""); continue
        for ln in textwrap.wrap(para, width=100, break_long_words=False):
            if font.size(ln)[0] <= max_w: out.append(ln)
            else: out.extend(textwrap.wrap(ln, width=80))
    return out

def make_prompt_text(delta: float|None, strike: bool) -> str:
    if delta is None:
        return ("Mission briefing:\n"
                "Feed Sentinel the BEST chunks to boost Recall and\n"
                "suppress FAR. Three bad chunks = round lost!")
    if strike:
        return ("Strike!\n"
                "That chunk eroded performance.\n"
                "Inspect Pos % and Δ hints before ingesting.")
    if delta < 0.02:
        return ("Minor tune-up.\n"
                "Telemetry nudged the model upwards.\n"
                "Hunt for bigger gains in the hint panel!")
    return ("Big boost!\n"
            "Sentinel just levelled-up noticeably.\n"
            "Keep the streak alive!")

def _ensure_score(uid):
    ref = get_firestore_db().collection("scoreboards").document(uid)
    snap = ref.get()
    if not snap.exists or "score" not in (snap.to_dict() or {}):
        ref.set({"score": 0}, merge=True)

try:
    from river.metrics.classification import Recall, Specificity
except ModuleNotFoundError:
    from river import metrics as _m
    Recall = _m.Recall
    class Specificity:
        bigger_is_better = True
        def __init__(self): self.tn = self.fp = 0
        def update(self, y, yp, **kw):
            if y == 0:
                self.tn += yp == 0
                self.fp += yp == 1
            return self
        def get(self):
            tot = self.tn + self.fp
            return self.tn / tot if tot else 0
        def works_with(self, _): return True

def compute_metrics(mgr, df_val) -> Dict[str, float]:
    lbl = _labelcol(df_val)
    rec, spec = Recall(), Specificity()
    for _, row in df_val.iterrows():
        y = int(row[lbl]) if not pd.isna(row[lbl]) else 0
        p = mgr.predict_one(row.drop(labels=[lbl]).to_dict()) or 0
        rec.update(y, p); spec.update(y, p)
    far = 1 - spec.get()
    return {"rec": rec.get(), "far": far, "score": rec.get()-far}

def analyse_chunk_preview(df_prev, mgr, val_stats):
    lbl = _labelcol(df_prev)
    hits = tp_new = fp_new = 0
    for _, row in df_prev.iterrows():
        y = int(row[lbl]) if not pd.isna(row[lbl]) else 0
        p = mgr.predict_one(row.drop(labels=[lbl]).to_dict()) or 0
        if p: hits += 1
        if y == 1 and p == 0: tp_new += 1
        elif y == 0 and p == 1: fp_new += 1
    pos_pct  = (df_prev[lbl] == 1).mean()
    hit_rate = hits/len(df_prev)
    drec     = tp_new / max(1, val_stats["n_pos"])
    dfar     = fp_new / max(1, val_stats["n_neg"])
    return {"pos_pct": pos_pct, "hit_rate": hit_rate, "gain": drec-dfar}

def pick_scenario(scr, uid) -> str|None:
    keys = list(SCENARIOS)
    per  = min(len(keys), (HEIGHT-170)//(BTN_H+GAP))
    frame_h = per*BTN_H + (per-1)*GAP + 40

    list_frame = pygame.Rect(60, 130,
                             LIST_W+SCROLL_GAP+SCROLL_W+40, frame_h)
    list_box   = pygame.Rect(list_frame.x+20, list_frame.y+20,
                             LIST_W, frame_h-40)

    help_w   = 480
    help_gap = 30
    help_rect = pygame.Rect(list_frame.right+help_gap,
                            list_frame.top, help_w, frame_h)
    help_pad = 20

    help_lines = [
        "Mission briefing:",
        "You are the SOC on night-shift.",
        "Leadership just unlocked a prototype ML add-on",
        "for your SIEM — codename ‘Sentinel’.",
        "",
        "Every CSV chunk is a training power-up.",
        "Feed Sentinel the right telemetry to",
        "LEVEL-UP its threat-spotting skills,",
        "but slip it junk chunks and Sentinel will",
        "explode with false alarms (strikes).",
        "",
        "",
        "How to play:",
        "• Model is scored on Recall − FAR.",
        "• Pick a chunk, press  Ingest.",
        "  Good chunk → + Score ; Bad → strike.",
        "• 3 strikes = round lost.",
        "• Cash-out anytime to keep points.",
        "",
        "Metrics:",
        "Recall (TPR) – attacks detected.",
        "FAR – false alarms.",
        "Score – Recall minus FAR."
    ]

    logout_rect = pygame.Rect(0,0,LOGOUT_W,LOGOUT_H)
    cancel_btn  = pygame.Rect(help_rect.centerx-95,
                              help_rect.bottom-70, 190, 60)

    scroll=dragging=y0=s0=0
    clock = pygame.time.Clock()
    while True:
        scr.fill(COLOR_BG)
        title = FONT_HUGE.render("Choose a Defender Scenario", True, COLOR_TXT)
        scr.blit(title, (WIDTH//2-title.get_width()//2, 70))

        draw_uid_chip(scr, uid, logout_rect)
        draw_panel(scr, list_frame)

        visible = keys[scroll:scroll+per]
        btns: List[Tuple[pygame.Rect,str]]=[]
        for i,k in enumerate(visible):
            r = pygame.Rect(list_box.x,
                            list_box.y+i*(BTN_H+GAP),
                            LIST_W, BTN_H)
            draw_button(scr, r, k, r.collidepoint(pygame.mouse.get_pos()))
            btns.append((r,k))

        thumb=None
        if len(keys)>per:
            track = pygame.Rect(list_box.right+SCROLL_GAP,
                                list_box.y, SCROLL_W, list_box.h)
            pygame.draw.rect(scr,(200,200,200),track)
            th=max(40,int(list_box.h*per/len(keys)))
            yof=int((list_box.h-th)*scroll/(len(keys)-per))
            thumb=pygame.Rect(track.x,track.y+yof,SCROLL_W,th)
            pygame.draw.rect(scr,(90,90,90),thumb)

        draw_panel(scr, help_rect)
        y = help_rect.y + help_pad
        for ln in help_lines:
            font = FONT_MED if ln.endswith(":") else FONT_HELP
            scr.blit(font.render(ln,True,COLOR_TXT),
                     (help_rect.x+help_pad,y))
            y += font.get_height()+2

        draw_button(scr, cancel_btn, "Cancel",
                    cancel_btn.collidepoint(pygame.mouse.get_pos()), FONT_MED)

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                if logout_rect.collidepoint(ev.pos): return "__logout__"
                if cancel_btn.collidepoint(ev.pos):  return "__cancel__"
                if thumb and thumb.collidepoint(ev.pos):
                    dragging=True; y0=ev.pos[1]; s0=scroll; continue
                for r,k in btns:
                    if r.collidepoint(ev.pos): return k

            if ev.type==pygame.MOUSEBUTTONUP and ev.button==1:
                dragging=False

            if ev.type==pygame.MOUSEMOTION and dragging and len(keys)>per:
                scroll=max(0,min(
                    int(s0+(ev.pos[1]-y0)*(len(keys)-per)/(list_box.h-th)),
                    len(keys)-per))

            if ev.type==pygame.MOUSEWHEEL:
                scroll=max(0,min(scroll-ev.y,len(keys)-per))

        pygame.display.flip(); clock.tick(30)

def play_round_ui(scr, key, uid) -> Tuple[float,int]|Tuple[str,int]:
    cfg   = SCENARIOS[key]
    urls  = list(cfg["chunks"])

    mgr      = get_model_manager(key, uid)   
    val_df   = get_csv(cfg["val"])
    val_stats = {
        "n_pos": int((val_df[_labelcol(val_df)]==1).sum()),
        "n_neg": int((val_df[_labelcol(val_df)]==0).sum())
    }

    previews = [get_csv(u, PREVIEW_N) for u in urls]
    def build_hints(): return [analyse_chunk_preview(df,mgr,val_stats)
                               for df in previews]
    hints = build_hints()

    base        = compute_metrics(mgr, val_df)
    current     = base.copy()
    last_score  = current["score"]         
    total_delta = strikes = 0

    prompt_lines = wrap_text(make_prompt_text(None,False),
                             FONT_HELP, 500-40)

    clock   = pygame.time.Clock()
    CARD_H  = 140
    per     = max(1,(HEIGHT-300)//(CARD_H+GAP))
    frame_h = per*CARD_H + (per-1)*GAP + 40
    frame   = pygame.Rect(60-20, 160-20,
                          LIST_W+SCROLL_GAP+SCROLL_W+40, frame_h)
    box     = pygame.Rect(frame.x+20, frame.y+20,
                          LIST_W, frame_h-40)

    score_card   = pygame.Rect(40, 70, 220, 60)
    strike_card  = pygame.Rect(score_card.right+20, 70, 180, 60)
    comment_card = pygame.Rect(660, 420, 500, 260)

    logout_rect = pygame.Rect(0,0,LOGOUT_W,LOGOUT_H)
    quit_rect   = pygame.Rect(WIDTH-110, 80, 90, 32)

    scroll = dragging = y0 = s0 = 0
    selected_idx: int|None = None

    while True:
        scr.fill(COLOR_BG)
        draw_uid_chip(scr, uid, logout_rect)
        draw_button(scr, quit_rect, "Quit",
                    quit_rect.collidepoint(pygame.mouse.get_pos()), FONT_SMALL)

        hdr = FONT_BIG.render(f"{key} – Round", True, COLOR_TXT)
        scr.blit(hdr, (WIDTH//2-hdr.get_width()//2, 30))

        draw_panel(scr, score_card)
        t = FONT_SMALL.render(f"Δ score: {total_delta:+.4f}", True, COLOR_TXT)
        scr.blit(t, (score_card.centerx-t.get_width()//2,
                     score_card.centery-t.get_height()//2))

        draw_panel(scr, strike_card)
        s = FONT_SMALL.render(f"Strikes: {strikes}/3", True, COLOR_TXT)
        scr.blit(s, (strike_card.centerx-s.get_width()//2,
                     strike_card.centery-s.get_height()//2))

        draw_panel(scr, frame)
        max_scroll = max(0,len(urls)-per)
        thumb=None
        if max_scroll:
            track=pygame.Rect(box.right+SCROLL_GAP, box.y, SCROLL_W, box.h)
            pygame.draw.rect(scr,(200,200,200),track)
            th=max(40,int(box.h*per/len(urls)))
            yof=int((box.h-th)*scroll/max_scroll)
            thumb=pygame.Rect(track.x,track.y+yof,SCROLL_W,th)
            pygame.draw.rect(scr,(90,90,90),thumb)

        visible = urls[scroll:scroll+per]
        cards: List[Tuple[pygame.Rect,int]]=[]
        for n,u in enumerate(visible):
            idx = urls.index(u)
            y   = box.y+n*(CARD_H+GAP)
            fill = COLOR_SELECTED_FILL if idx==selected_idx else COLOR_CARD
            card = pygame.Rect(box.x, y, LIST_W, CARD_H)
            draw_panel(scr, card, fill=fill)
            cards.append((card,idx))

            h = hints[idx]
            scr.blit(FONT_SMALL.render(f"Chunk {idx+1}",True,COLOR_TXT),
                     (card.x+20, card.y+10))
            pygame.draw.rect(scr,(40,80,40),
                             pygame.Rect(card.x+20, card.y+34, 100,8),1)
            bw=int(100*h["pos_pct"])
            pygame.draw.rect(scr,(90,140,90),
                             pygame.Rect(card.x+20, card.y+34, bw,8),
                             border_radius=4)
            scr.blit(FONT_SMALL.render(
                f"Pos {h['pos_pct']*100:4.1f}% · Hit {h['hit_rate']*100:4.1f}%",
                True, COLOR_TXT),
                (card.x+140, card.y+28))
            scr.blit(FONT_SMALL.render(f"Δ = {h['gain']:+.3f}",True,COLOR_TXT),
                     (card.x+140, card.y+54))

        m_card = pygame.Rect(660,120,500,260)
        draw_panel(scr, m_card)
        scr.blit(FONT_MED.render("Current Stats",True,COLOR_TXT),
                 (m_card.x+20, m_card.y+20))
        scr.blit(FONT_SMALL.render(f"Recall (TPR): {current['rec']:.3f}",
                                   True,COLOR_TXT),
                 (m_card.x+40, m_card.y+80))
        scr.blit(FONT_SMALL.render(f"FAR: {current['far']:.3f}",
                                   True,COLOR_TXT),
                 (m_card.x+40, m_card.y+110))
        scr.blit(FONT_SMALL.render(f"Score(T-F): {current['score']:.3f}",
                                   True,COLOR_TXT),
                 (m_card.x+40, m_card.y+140))

        draw_panel(scr, comment_card)
        yy = comment_card.y+20
        for ln in prompt_lines:
            scr.blit(FONT_HELP.render(ln,True,COLOR_TXT),
                     (comment_card.x+20,yy))
            yy+=FONT_HELP.get_height()+2

        btn_w, btn_h, gap = 190,60,40
        cash_btn_x   = comment_card.centerx - btn_w - gap//2
        ingest_btn_x = comment_card.centerx + gap//2
        btn_y        = HEIGHT-90
        cash_btn   = pygame.Rect(cash_btn_x, btn_y, btn_w, btn_h)
        ingest_btn = pygame.Rect(ingest_btn_x, btn_y, btn_w, btn_h)

        draw_button(scr, cash_btn, "Cash Out",
                    cash_btn.collidepoint(pygame.mouse.get_pos()), FONT_BIG)
        draw_button(scr, ingest_btn, "Ingest",
                    ingest_btn.collidepoint(pygame.mouse.get_pos()), FONT_BIG)

        pygame.display.flip(); clock.tick(30)

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                if logout_rect.collidepoint(ev.pos):
                    return "__logout__", strikes
                if quit_rect.collidepoint(ev.pos):
                    return "__quit__", strikes
                if thumb and thumb.collidepoint(ev.pos):
                    dragging=True; y0=ev.pos[1]; s0=scroll; continue
                for card,idx in cards:
                    if card.collidepoint(ev.pos):
                        selected_idx = idx if selected_idx!=idx else None
                        break
                if cash_btn.collidepoint(ev.pos):
                    return total_delta, strikes

                if ingest_btn.collidepoint(ev.pos) and selected_idx is not None:
                    chunk_url = urls[selected_idx]

                    try:
                        res = backend_ingest(key, chunk_url)
                    except Exception as exc:
                        print("[ERROR] backend call failed:", exc)
                        continue
                    score_backend = res["value"] - res["far"]
                    delta = score_backend - last_score
                    last_score = score_backend
                    strike_now = delta <= 0
                    if strike_now: strikes += 1
                    else: total_delta += delta
                    prompt_lines = wrap_text(
                        make_prompt_text(delta, strike_now),
                        FONT_HELP, comment_card.width-40)

                    df  = get_csv(chunk_url)
                    lbl = _labelcol(df)
                    for _, row in df.iterrows():
                        truth = int(row[lbl]) if not pd.isna(row[lbl]) else 0
                        mgr.update_one(row.drop(labels=[lbl]).to_dict(), truth)
                    mgr.save()
                    current = compute_metrics(mgr, val_df)

                    urls.pop(selected_idx); previews.pop(selected_idx)
                    selected_idx=None; hints=build_hints()
                    scroll=min(scroll,max(0,len(urls)-per))

            if ev.type==pygame.MOUSEBUTTONUP and ev.button==1:
                dragging=False

            if ev.type==pygame.MOUSEMOTION and dragging and max_scroll:
                dy=ev.pos[1]-y0
                scroll=max(0,min(int(s0+dy*max_scroll/(box.h-thumb.h)),
                                 max_scroll))

            if ev.type==pygame.MOUSEWHEEL and max_scroll:
                scroll=max(0,min(scroll-ev.y,max_scroll))

        if not urls or strikes>=3:
            return (0.0 if strikes>=3 else total_delta, strikes)

def summary_splash(scr, scenario, gained, strikes, pts, total):
    scr.fill(COLOR_BG)
    card = pygame.Rect(260,230,760,340)
    draw_panel(scr, card)
    y=card.y+45
    for ln in (
        f"Scenario : {scenario}",
        f"Strikes   : {strikes}/3",
        f"Δ Score   : {gained:+.4f}",
        f"+ Points  : {pts}",
        f"Total     : {total}",
        "",
        "Closing automatically…"
    ):
        t=FONT_MED.render(ln,True,COLOR_TXT)
        scr.blit(t,(card.x+60,y)); y+=55
    pygame.display.flip()
    pygame.time.wait(15000)

def run_defender_ui(uid="demo_user"):
    _ensure_score(uid)
    scr = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("DEFENDER – Recall/FAR edition")

    while True:
        choice = pick_scenario(scr, uid)
        if choice in ("__logout__","__cancel__",None):
            return "menu" if choice=="__cancel__" else choice

        scenario_key = choice
        gained, strikes = play_round_ui(scr, scenario_key, uid)
        if gained in ("__logout__","__quit__"):
            continue

        pts = 0
        if isinstance(gained,float) and gained>0:
            full_pts = delta_to_points(gained,
                                       SCENARIOS[scenario_key]["difficulty"])
            pts = full_pts//2
            update_user_score(uid, pts)
            add_defender_score(uid, scenario_key, pts)

        total = get_user_score(uid)
        summary_splash(scr, scenario_key, gained, strikes, pts, total)


if __name__ == "__main__":
    os.environ.setdefault("CYBER_ML_API_BASE",
        "URL")
    try:
        run_defender_ui("demo_user")
    except KeyboardInterrupt:
        print("\nbye")
