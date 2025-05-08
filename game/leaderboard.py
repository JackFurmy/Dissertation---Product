
from __future__ import annotations
import os
import sys
import time
import pygame
from typing import Dict, List, Tuple

from firebase_init import get_firestore_db
from user_session import (
    _ensure_attacker_score_fields,
    _ensure_defender_score_fields,
)

def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel_path)

WIDTH, HEIGHT = 1200, 800
pygame.init()

FONT_BIG   = pygame.font.SysFont(None, 40)
FONT_MED   = pygame.font.SysFont(None, 30)
FONT_SMALL = pygame.font.SysFont(None, 24)
FONT_UID   = pygame.font.SysFont(None, 22)

COLOR_BG        = (245, 255, 246)
COLOR_CARD      = (233, 239, 232)
COLOR_TXT       = (0,   0,   0)
COLOR_BTN       = (132, 145, 163)
COLOR_HOV       = (150, 163, 180)
COLOR_UID_CHIP  = (240, 248, 235)
COLOR_UID_EDGE  = (200, 208, 195)

LIST_W, SCROLL_W, GAP = 560, 14, 8
REFRESH_SECONDS        = 10
TOPBAR_H               = 66
ROW_H                  = 38

CHIP_H, CHIP_RADIUS    = 36, 14
LOGOUT_W, LOGOUT_H     = 100, 36
CHIP_MARGIN            = 20


def _shade(rgb, d):
    return tuple(max(0, min(255, c + d)) for c in rgb)

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

def draw_button(surf, rect, txt, hover=False, font=FONT_MED):
    fill = COLOR_HOV if hover else COLOR_BTN
    pygame.draw.rect(surf, fill, rect, border_radius=8)
    t = font.render(txt, True, (255, 255, 255))
    surf.blit(t, (rect.centerx - t.get_width()//2,
                  rect.centery - t.get_height()//2))

def draw_uid_chip(surf, uid, logout_rect: pygame.Rect):
    uid_surf = FONT_UID.render(f"UID: {uid}", True, COLOR_TXT)
    chip_w   = uid_surf.get_width() + LOGOUT_W + CHIP_MARGIN*3
    chip_rect = pygame.Rect(WIDTH - chip_w - 20, CHIP_MARGIN,
                            chip_w, CHIP_H)

    pygame.draw.rect(surf, COLOR_UID_CHIP, chip_rect, border_radius=CHIP_RADIUS)
    pygame.draw.rect(surf, COLOR_UID_EDGE, chip_rect, 1, border_radius=CHIP_RADIUS)
    surf.blit(uid_surf, (chip_rect.x + CHIP_MARGIN,
                         chip_rect.centery - uid_surf.get_height()//2))

    logout_rect.size = (LOGOUT_W, LOGOUT_H)
    logout_rect.topleft = (chip_rect.right - LOGOUT_W - CHIP_MARGIN, CHIP_MARGIN)
    pygame.draw.rect(surf, COLOR_UID_CHIP, logout_rect, border_radius=CHIP_RADIUS)
    pygame.draw.rect(surf, COLOR_UID_EDGE, logout_rect, 1, border_radius=CHIP_RADIUS)
    lo = FONT_UID.render("Log Out", True, COLOR_TXT)
    surf.blit(lo, (logout_rect.centerx - lo.get_width()//2,
                   logout_rect.centery - lo.get_height()//2))

def fetch_scoreboards() -> List[dict]:
    db = get_firestore_db()
    docs = db.collection("scoreboards").stream()
    return [{**d.to_dict(), "uid": d.id} for d in docs]

def build_lines(raw: List[dict], mode: str,
                sub_key: str | None) -> List[Tuple[str,int]]:
    rows: List[Tuple[str,int]] = []
    for doc in raw:
        uid = doc["uid"]

        if mode == "total":
            rows.append((uid, doc.get("score", 0)))

        elif mode == "attacker":
            a = _ensure_attacker_score_fields(doc)
            rows.append((uid, a["attacker_scores"]["overall"]))

        elif mode == "defender":
            d = _ensure_defender_score_fields(doc)
            rows.append((uid, d["defender_scores"]["overall"]))

        elif mode == "dataset":
            a = _ensure_attacker_score_fields(doc)
            rows.append((uid, a["attacker_scores"]["datasets"].get(sub_key, 0)))

        elif mode == "category":
            d   = _ensure_defender_score_fields(doc)
            cat = d["defender_scores"]["categories"]
            rows.append((uid, cat.get(sub_key, {}).get("overall", 0)))

        elif mode == "scenario":
            d   = _ensure_defender_score_fields(doc)
            cat = d["defender_scores"]["categories"]
            if sub_key and "/" in sub_key:
                c, sc = sub_key.split("/", 1)
                pts   = cat.get(c, {}).get("datasets", {}).get(sc, 0)
            else:
                pts = 0
            rows.append((uid, pts))

    return sorted(rows, key=lambda t: t[1], reverse=True)

FILTERS       = ["total", "attacker", "defender"]
DEFENDER_CATS = ["phishing", "iot", "linux", "windows"]

def run_leaderboard_ui(uid: str) -> str:
    scr = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Global Leaderboard")

    top_idx, sub_key, cat_expanded = 0, None, None
    list_scroll = side_scroll = 0
    drag_main = drag_side = False
    y0 = s0 = 0

    DATASETS: List[str] = []
    CAT_DS_MAP: Dict[str, List[str]] = {}
    last_fetch, raw_docs = 0, []

    logout_rect = pygame.Rect(0, 0, LOGOUT_W, LOGOUT_H)

    clock = pygame.time.Clock()
    while True:

        if time.time() - last_fetch > REFRESH_SECONDS:
            raw_docs   = fetch_scoreboards()
            last_fetch = time.time()

            ds_set, cat_map = set(), {}
            for d in raw_docs:
                att = _ensure_attacker_score_fields(d)
                ds_set.update(att["attacker_scores"]["datasets"].keys())
                de  = _ensure_defender_score_fields(d)
                for cat, blk in de["defender_scores"]["categories"].items():
                    cat_map.setdefault(cat, set()).update(blk["datasets"].keys())

            DATASETS   = sorted(ds_set)
            CAT_DS_MAP = {c: sorted(v) for c,v in cat_map.items()}

            if FILTERS[top_idx] == "dataset" and sub_key not in DATASETS:
                sub_key = DATASETS[0] if DATASETS else None
            if FILTERS[top_idx] == "category" and sub_key not in DEFENDER_CATS:
                sub_key = None
            if FILTERS[top_idx] == "scenario" and sub_key not in [
                f"{c}/{s}" for c in CAT_DS_MAP for s in CAT_DS_MAP[c]
            ]:
                sub_key = None

        cur_filter = FILTERS[top_idx]
        if cur_filter == "total":
            mode = "total"
        elif cur_filter == "attacker":
            mode = "dataset" if sub_key else "attacker"
        else:  
            if sub_key and "/" in (sub_key or ""):
                mode = "scenario"
            elif sub_key:
                mode = "category"
            else:
                mode = "defender"

        lines = build_lines(raw_docs, mode, sub_key)

        top_panel   = pygame.Rect(40, 78, LIST_W, TOPBAR_H)
        btn_w       = top_panel.w // 3 - 10
        btn_tot     = pygame.Rect(top_panel.x+5,           top_panel.y+8,
                                  btn_w, top_panel.h-16)
        btn_att     = pygame.Rect(btn_tot.right+5,         top_panel.y+8,
                                  btn_w, top_panel.h-16)
        btn_def     = pygame.Rect(btn_att.right+5,         top_panel.y+8,
                                  btn_w, top_panel.h-16)

        list_frame  = pygame.Rect(40, top_panel.bottom+10,
                                  LIST_W+40, HEIGHT-top_panel.bottom-120)
        list_box    = pygame.Rect(list_frame.x+20, list_frame.y+20,
                                  LIST_W, list_frame.h-40)

        side_frame  = pygame.Rect(list_frame.right+40, list_frame.y,
                                  WIDTH-list_frame.right-60,
                                  list_frame.h)

        quit_btn    = pygame.Rect(WIDTH-140, HEIGHT-58, 104, 42)

        per_rows     = max(1, list_box.h // ROW_H)
        max_main_scr = max(0, len(lines) - per_rows)

        if cur_filter == "attacker":
            side_items = DATASETS
        elif cur_filter == "defender" and not sub_key:
            side_items = DEFENDER_CATS
        elif cur_filter == "defender" and sub_key and "/" not in sub_key:
            side_items = CAT_DS_MAP.get(sub_key, [])
        else:
            side_items = []

        per_side     = max(1, (side_frame.h-40)//46)
        max_side_scr = max(0, len(side_items)-per_side)

        scr.fill(COLOR_BG)
        scr.blit(FONT_BIG.render("Global Leaderboard", True, COLOR_TXT),
                 (list_frame.x, 26))

        draw_uid_chip(scr, uid, logout_rect)

        draw_panel(scr, top_panel)
        draw_button(scr, btn_tot, "Totals",   top_idx==0)
        draw_button(scr, btn_att, "Attacker", top_idx==1)
        draw_button(scr, btn_def, "Defender", top_idx==2)

        draw_panel(scr, list_frame)
        visible = lines[list_scroll:list_scroll+per_rows]
        for i,(u,p) in enumerate(visible):
            y = list_box.y + i*ROW_H
            row = pygame.Rect(list_box.x, y, list_box.w, ROW_H-4)
            pygame.draw.rect(scr, COLOR_CARD, row)
            pygame.draw.rect(scr, COLOR_TXT,  row, 1)
            scr.blit(FONT_SMALL.render(f"{list_scroll+i+1}.", True, COLOR_TXT),
                     (row.x+8, y+6))
            scr.blit(FONT_SMALL.render(u, True, COLOR_TXT),
                     (row.x+40, y+6))
            pts_surf = FONT_SMALL.render(str(p), True, COLOR_TXT)
            scr.blit(pts_surf, (row.right-pts_surf.get_width()-14, y+6))

        thumb_main = None
        if max_main_scr:
            track = pygame.Rect(list_box.right+GAP, list_box.y,
                                SCROLL_W, list_box.h)
            pygame.draw.rect(scr, (200,200,200), track)
            th   = max(40, int(track.h*per_rows/len(lines)))
            yof  = int((track.h-th)*list_scroll/max_main_scr)
            thumb_main = pygame.Rect(track.x, track.y+yof, SCROLL_W, th)
            pygame.draw.rect(scr, (90,90,90), thumb_main)

        draw_panel(scr, side_frame)
        inner = pygame.Rect(side_frame.x+20, side_frame.y+20,
                            side_frame.w-40, side_frame.h-40)

        if side_items:
            vis_side = side_items[side_scroll:side_scroll+per_side]
            for i,lab in enumerate(vis_side):
                y = inner.y + i*46
                btn = pygame.Rect(inner.x, y, inner.w, 40)
                draw_button(scr, btn, lab, lab==sub_key, FONT_SMALL)

            if max_side_scr:
                track = pygame.Rect(inner.right+GAP, inner.y,
                                    SCROLL_W, inner.h)
                pygame.draw.rect(scr, (200,200,200), track)
                th   = max(40, int(track.h*per_side/len(side_items)))
                yof  = int((track.h-th)*side_scroll/max_side_scr)
                thumb_side = pygame.Rect(track.x, track.y+yof,
                                         SCROLL_W, th)
                pygame.draw.rect(scr, (90,90,90), thumb_side)
        else:
            msg = FONT_SMALL.render("No data", True, COLOR_TXT)
            scr.blit(msg, (inner.centerx-msg.get_width()//2,
                           inner.centery-msg.get_height()//2))

        draw_button(scr, quit_btn, "Quit",
                    quit_btn.collidepoint(pygame.mouse.get_pos()))

        pygame.display.flip()
        clock.tick(30)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if quit_btn.collidepoint(e.pos):
                    return "menu"
                if logout_rect.collidepoint(e.pos):
                    return "logout"

                if btn_tot.collidepoint(e.pos):
                    top_idx, sub_key, cat_expanded = 0, None, None
                    list_scroll = side_scroll = 0
                elif btn_att.collidepoint(e.pos):
                    top_idx, sub_key, cat_expanded = 1, None, None
                    list_scroll = side_scroll = 0
                elif btn_def.collidepoint(e.pos):
                    top_idx, sub_key, cat_expanded = 2, None, None
                    list_scroll = side_scroll = 0

                if inner.collidepoint(e.pos) and side_items:
                    idx = (e.pos[1]-inner.y)//46 + side_scroll
                    if 0 <= idx < len(side_items):
                        label = side_items[idx]
                        if cur_filter == "attacker":
                            sub_key = label
                        elif cur_filter == "defender":
                            if label in DEFENDER_CATS:
                                if sub_key == label and cat_expanded:
                                    sub_key, cat_expanded = None, None
                                else:
                                    sub_key, cat_expanded = label, True
                                    side_scroll = 0
                            else:
                                sub_key = f"{sub_key}/{label}" if sub_key else label
                        list_scroll = 0

                if thumb_main and thumb_main.collidepoint(e.pos):
                    drag_main, y0, s0 = True, e.pos[1], list_scroll
                if 'thumb_side' in locals() and \
                   thumb_side.collidepoint(e.pos):
                    drag_side, y0, s0 = True, e.pos[1], side_scroll

            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                drag_main = drag_side = False

            if e.type == pygame.MOUSEMOTION:
                if drag_main and max_main_scr:
                    dy = e.pos[1]-y0
                    list_scroll = max(0, min(int(s0 + dy*max_main_scr/
                                                 (list_box.h-thumb_main.height)),
                                             max_main_scr))
                if drag_side and max_side_scr:
                    dy = e.pos[1]-y0
                    side_scroll = max(0, min(int(s0 + dy*max_side_scr/
                                                 (inner.h-thumb_side.height)),
                                             max_side_scr))

            if e.type == pygame.MOUSEWHEEL:
                if list_frame.collidepoint(pygame.mouse.get_pos()) and max_main_scr:
                    list_scroll = max(0, min(list_scroll-e.y, max_main_scr))
                elif side_frame.collidepoint(pygame.mouse.get_pos()) and max_side_scr:
                    side_scroll = max(0, min(side_scroll-e.y, max_side_scr))

if __name__ == "__main__":
    res = run_leaderboard_ui("demo_user")
    pygame.quit()
    print("Return:", res)
