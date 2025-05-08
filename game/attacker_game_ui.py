import os, sys, textwrap, tempfile, time
import pygame
import matplotlib; matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import requests

from user_session import update_user_score, add_attacker_score
from ml_integration import (
    load_baseline_model,
    fetch_dataset_from_firebase,
    fetch_dataset_via_url,
)
from firebase_init import get_firestore_db



try:
    from util_paths import resource_path
except Exception:
    def resource_path(rel_path: str) -> str:
        base = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
        return os.path.join(base, rel_path)


API_BASE = os.getenv(
    "CYBER_ML_API_BASE",
    "URL"
).rstrip("/")

def call_backend(dataset: str, removed_cols: list[str]) -> dict:
    r = requests.post(
        f"{API_BASE}/attacker/round",
        json={"dataset": dataset, "removed_cols": removed_cols},
        timeout=900
    )
    r.raise_for_status()
    return r.json()



pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800

FONT_LARGE  = pygame.font.SysFont(None, 48)
FONT_MED    = pygame.font.SysFont(None, 36)
FONT_SMALL  = pygame.font.SysFont(None, 28)
FONT_INFO   = pygame.font.SysFont(None, 24)
FONT_UID    = pygame.font.SysFont(None, 24)

COLOR_BG,  COLOR_TEXT = (247, 255, 246), (0, 0, 0)
COLOR_BTN, COLOR_HOV  = (132, 145, 163), (150, 163, 180)
COLOR_CARD            = (233, 239, 232)
COLOR_WHITE           = (255, 255, 255)

BUTTON_W, BUTTON_H   = 220, 64
COL_BOX_W, COL_BOX_H = 400, 500
ROUNDS               = 5
GENERIC_MAX_PICK     = 15

CHIP_COLOR   = (240, 248, 235)
CHIP_OUTLINE = (200, 208, 195)
CHIP_H, CHIP_R = 36, 18
CHIP_MARGIN  = 20

LOGOUT_W, LOGOUT_H = 100, 36

AVAILABLE_DATASETS = [
    "cicids2017", "unsw_nb15", "ctu13",
    "iot23", "ember", "malware"
]

MI_IMAGE_DIR = resource_path("mi figure")

DATASET_TO_IMAGE = {
    "cicids2017": os.path.join(MI_IMAGE_DIR, "cicids2017_mi.png"),
    "unsw_nb15" : os.path.join(MI_IMAGE_DIR, "unsw_nb15_mi.png"),
    "ctu13"     : os.path.join(MI_IMAGE_DIR, "ctu13_mi.png"),
    "iot23"     : os.path.join(MI_IMAGE_DIR, "iot23_mi.png"),
    "ember"     : os.path.join(MI_IMAGE_DIR, "ember_mi.png"),
    "malware"   : os.path.join(MI_IMAGE_DIR, "malware_mi.png"),
}
TMP_DIR = tempfile.gettempdir()

CICIDS_PICK_LIMITS  = [9, 9, 9, 9, 9];  CICIDS_POINTS_FACTOR  = 5
UNSW_PICK_LIMITS    = [4, 4, 5, 5, 6];  UNSW_POINTS_FACTOR    = 8
CTU_PICK_LIMITS     = [1, 1, 1, 2, 3];  CTU_POINTS_FACTOR     = 10
IOT23_PICK_LIMITS   = [3, 3, 3, 4, 4];  IOT23_POINTS_FACTOR   = 9
EMBER_PICK_LIMITS   = [1, 1, 1, 1, 1];  EMBER_POINTS_FACTOR   = 3
MALWARE_PICK_LIMITS = [3, 5, 6, 6, 7];  MALWARE_POINTS_FACTOR = 3
MAX_SAMPLE_ROWS     = 500_000


def _shade(rgb, delta):
    return tuple(max(0, min(255, c + delta)) for c in rgb)

def draw_button_3d(surf, rect, label, hover=False, font=FONT_MED):
    fill = COLOR_HOV if hover else COLOR_BTN
    pygame.draw.rect(surf, fill, rect, border_radius=8)
    pygame.draw.line(surf, _shade(fill, 40),
                     (rect.left+1, rect.top+1), (rect.right-2, rect.top+1), 2)
    pygame.draw.line(surf, _shade(fill, 40),
                     (rect.left+1, rect.top+1), (rect.left+1, rect.bottom-2), 2)
    pygame.draw.line(surf, _shade(fill, -40),
                     (rect.left+1, rect.bottom-2), (rect.right-2, rect.bottom-2), 2)
    pygame.draw.line(surf, _shade(fill, -40),
                     (rect.right-2, rect.top+1), (rect.right-2, rect.bottom-2), 2)
    txt = font.render(label, True, COLOR_WHITE)
    surf.blit(txt, (rect.centerx - txt.get_width()//2,
                    rect.centery - txt.get_height()//2))

def draw_panel_3d(surf, rect, fill=COLOR_CARD):
    pygame.draw.rect(surf, fill, rect, border_radius=10)
    pygame.draw.line(surf, _shade(fill, 45),
                     (rect.left+1, rect.top+1), (rect.right-2, rect.top+1), 2)
    pygame.draw.line(surf, _shade(fill, 45),
                     (rect.left+1, rect.top+1), (rect.left+1, rect.bottom-2), 2)
    pygame.draw.line(surf, _shade(fill, -45),
                     (rect.left+1, rect.bottom-2), (rect.right-2, rect.bottom-2), 2)
    pygame.draw.line(surf, _shade(fill, -45),
                     (rect.right-2, rect.top+1), (rect.right-2, rect.bottom-2), 2)

def draw_scrollbar(surf, box, total, visible, offset):
    if total <= visible:
        return None
    track_w, margin = 12, 10
    track = pygame.Rect(box.right-track_w-2, box.y+margin,
                        track_w, box.height-2*margin)
    pygame.draw.rect(surf, (200,200,200), track)
    thumb_h = int(track.height * (visible / total))
    thumb_y = track.y + int((track.height-thumb_h)
                            * (offset / max(1, total-visible)))
    thumb = pygame.Rect(track.x, thumb_y, track_w, thumb_h)
    pygame.draw.rect(surf, (120,120,120), thumb)
    return thumb

def handle_scroll_click(box, total, visible, offset, event):
    if total <= visible or event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
        return offset
    track_w, margin = 12, 10
    track = pygame.Rect(box.right-track_w-2, box.y+margin,
                        track_w, box.height-2*margin)
    if track.collidepoint(event.pos):
        rel = (event.pos[1]-track.y) / max(1, track.height-int(track.height*(visible/total)))
        return int(rel * (total - visible))
    return offset

def wrap_text(txt, font, max_w):
    out = []
    for para in txt.split("\n"):
        if not para:
            out.append("")
            continue
        for line in textwrap.wrap(para, width=100, break_long_words=False):
            out.append(line) if font.size(line)[0] <= max_w \
                else out.extend(textwrap.wrap(line, width=80))
    return out


def draw_uid_chip(surface, uid_text, logout_rect=None):
    uid_surf = FONT_UID.render(f"UID: {uid_text}", True, COLOR_TEXT)
    chip_w = (uid_surf.get_width() + 24 if logout_rect is None
              else uid_surf.get_width() + logout_rect.width + 36)
    chip_x = SCREEN_WIDTH - chip_w - CHIP_MARGIN
    chip_y = CHIP_MARGIN
    chip_rect = pygame.Rect(chip_x, chip_y, chip_w, CHIP_H)
    pygame.draw.rect(surface, CHIP_COLOR, chip_rect, border_radius=CHIP_R)
    pygame.draw.rect(surface, CHIP_OUTLINE, chip_rect, 1, border_radius=CHIP_R)
    surface.blit(uid_surf, (chip_x + 12,
                            chip_y + (CHIP_H - uid_surf.get_height())//2))
    if logout_rect:
        logout_rect.y = chip_y
        logout_rect.x = chip_rect.right - logout_rect.width - 12
        pygame.draw.rect(surface, CHIP_COLOR, logout_rect, border_radius=CHIP_R)
        pygame.draw.rect(surface, CHIP_OUTLINE, logout_rect, 1, border_radius=CHIP_R)
        lo_txt = FONT_UID.render("Log Out", True, COLOR_TEXT)
        surface.blit(lo_txt, (logout_rect.centerx - lo_txt.get_width()//2,
                              logout_rect.centery - lo_txt.get_height()//2))
    return chip_rect


def regenerate_mi_chart(df, dataset, rnd):
    if "label" not in df.columns or df["label"].nunique() < 2:
        return
    df = (df.sample(frac=.25 if dataset == "iot23" else .5, random_state=42)
          if len(df) > 2 else df)
    if len(df) > MAX_SAMPLE_ROWS:
        df = df.sample(n=MAX_SAMPLE_ROWS, random_state=42)
    X, y = df.drop(columns="label"), df["label"]
    try:
        mi = mutual_info_classif(X, y, random_state=42, discrete_features="auto")
    except Exception as e:
        print("[WARN] MI calc failed:", e); return
    order = mi.argsort()[::-1][:20]
    plt.figure(figsize=(10,4)); plt.bar(X.columns[order], mi[order])
    plt.xticks(rotation=90, fontsize=6); plt.tight_layout()
    fn = f"{dataset}_round{rnd}.png"
    out_path = os.path.join(TMP_DIR, fn)
    plt.savefig(out_path, dpi=120); plt.close()
    return out_path


def _show_splash(scr, title, lines):
    scr.fill(COLOR_BG)
    card_w, card_h = 600, 340
    card = pygame.Rect((SCREEN_WIDTH-card_w)//2,
                       (SCREEN_HEIGHT-card_h)//2,
                       card_w, card_h)
    head = pygame.Rect(card.left, card.top, card.w, 70)
    draw_panel_3d(scr, card, COLOR_CARD)
    pygame.draw.rect(scr, COLOR_BTN, head, 0,
                     border_top_left_radius=10, border_top_right_radius=10)
    pygame.draw.rect(scr, _shade(COLOR_BTN, -40), head, 2,
                     border_top_left_radius=10, border_top_right_radius=10)
    scr.blit(FONT_LARGE.render(title, True, COLOR_WHITE),
             (head.centerx-FONT_LARGE.size(title)[0]//2,
              head.centery-FONT_LARGE.get_height()//2))
    y = head.bottom + 25
    for ln in lines:
        scr.blit(FONT_MED.render(ln, True, COLOR_TEXT),
                 (card.left+30, y))
        y += FONT_MED.get_height() + 8
    pygame.display.flip()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return

def splash_cicids(s):  _show_splash(s, "CICIDS-2017 – Rules",
    ["Round pick limits: 9 • 9 • 9 • 9 • 9",
     "Scoring: 5 pts / 0.10 pp reduced", "Press Enter To Start…"])
def splash_unsw(s):    _show_splash(s, "UNSW-NB15 – Rules",
    ["Round pick limits: 4 • 4 • 5 • 5 • 6",
     "Scoring: 8 pts / 0.10 pp reduced", "Press Enter To Start…"])
def splash_ctu13(s):   _show_splash(s, "CTU-13 – Rules",
    ["Round pick limits: 1 • 1 • 1 • 2 • 3",
     "Scoring: 10 pts / 0.10 pp reduced", "Press Enter To Start…"])
def splash_iot23(s):   _show_splash(s, "IoT-23 – Rules",
    ["Round pick limits: 3 • 3 • 3 • 4 • 4",
     "Scoring: 9 pts / 0.10 pp reduced", "Press Enter To Start…"])
def splash_ember(s):   _show_splash(s, "EMBER – Rules",
    ["Round pick limits: 1 • 1 • 1 • 1 • 1",
     "Scoring: 3 pts / 0.10 pp reduced", "Press Enter To Start…"])
def splash_malware(s): _show_splash(s, "Malware – Rules",
    ["Round pick limits: 3 • 5 • 6 • 6 • 7",
     "Scoring: 3 pts / 0.10 pp reduced", "Press Enter To Start…"])


def summary_splash(scr, ds_name, metric_dict, points_total):
    title = f"{ds_name.upper()} – Results"
    lines = [
        f"Total points: {points_total}",
        "",
        "Final metrics:",
        f"accuracy : {metric_dict.get('accuracy', 0):.4f}",
        f"macro_f1 : {metric_dict.get('macro_f1', 0):.4f}",
        f"roc_auc  : {metric_dict.get('roc_auc', 0):.4f}",
        f"pr_auc   : {metric_dict.get('pr_auc', 0):.4f}",
        f"tnr      : {metric_dict.get('tnr', 0):.4f}",
        f"far      : {metric_dict.get('far', 0):.4f}",
        "",
        "This window will close automatically…",
    ]
    scr.fill(COLOR_BG)
    card_w, card_h = 600, 380
    card = pygame.Rect((SCREEN_WIDTH-card_w)//2,
                       (SCREEN_HEIGHT-card_h)//2,
                       card_w, card_h)
    head = pygame.Rect(card.left, card.top, card.w, 70)
    draw_panel_3d(scr, card, COLOR_CARD)
    pygame.draw.rect(scr, COLOR_BTN, head, 0,
                     border_top_left_radius=10, border_top_right_radius=10)
    pygame.draw.rect(scr, _shade(COLOR_BTN, -40), head, 2,
                     border_top_left_radius=10, border_top_right_radius=10)
    scr.blit(FONT_LARGE.render(title, True, COLOR_WHITE),
             (head.centerx-FONT_LARGE.size(title)[0]//2,
              head.centery-FONT_LARGE.get_height()//2))
    y = head.bottom + 25
    for ln in lines:
        scr.blit(FONT_MED.render(ln, True, COLOR_TEXT),
                 (card.left+30, y))
        y += FONT_MED.get_height() + 8
    pygame.display.flip()
    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < 15000:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return
        pygame.time.delay(30)


def pick_dataset_ui(screen, uid):
    clock = pygame.time.Clock()
    left, gap, y0 = 100, 20, 150
    btn_rects = [pygame.Rect(left, y0 + i * (BUTTON_H + gap),
                             BUTTON_W, BUTTON_H)
                 for i in range(len(AVAILABLE_DATASETS))]
    cancel = pygame.Rect(left,
                         y0 + len(btn_rects) * (BUTTON_H + gap),
                         BUTTON_W, BUTTON_H)
    frame = pygame.Rect(left - 20, y0 - 20,
                        BUTTON_W + 40,
                        cancel.bottom - y0 + 50)

    info_y      = max(60, frame.top - 10)
    info_height = min(frame.height + 20, SCREEN_HEIGHT - info_y - 40)
    info_panel  = pygame.Rect(frame.right + 30, info_y,
                              SCREEN_WIDTH - frame.right - 60, info_height)

    title = FONT_LARGE.render("Pick a Scenario Dataset to Attack", True, COLOR_TEXT)

    info_text = """Welcome, attacker!

Goal:
Reduce the baseline model's accuracy
by removing feature columns.

Metrics you’ll see every round:

• accuracy – out of all predictions, how many were right?
• macro F1 – recall & precision balance, averaged across classes
• ROC AUC – separates attack from benign, threshold-free
• PR AUC  – like ROC AUC but focuses on the attack class
• TNR     – % of normal traffic left untouched
• FAR     – % of normal traffic flagged as attack

Choose a scenario on the left to begin!













NOTE: If you click the columns in the game and they don't go green please click off the application then back on (This is a known pygame bug)"""
    info_lines = wrap_text(info_text, FONT_INFO, info_panel.width - 40)
    logout_btn = pygame.Rect(0, 0, LOGOUT_W, LOGOUT_H)

    while True:
        screen.fill(COLOR_BG)
        mpos = pygame.mouse.get_pos()
        screen.blit(title, (left, 60))
        draw_uid_chip(screen, uid_text=uid, logout_rect=logout_btn)
        draw_panel_3d(screen, frame, _shade(COLOR_BTN, 60))
        for rect, name in zip(btn_rects, AVAILABLE_DATASETS):
            draw_button_3d(screen, rect, name, rect.collidepoint(mpos))
        draw_button_3d(screen, cancel, "Cancel", cancel.collidepoint(mpos))
        draw_panel_3d(screen, info_panel, COLOR_CARD)
        yy = info_panel.y + 20
        for ln in info_lines:
            if yy > info_panel.bottom - 20:
                break
            screen.blit(FONT_INFO.render(ln, True, COLOR_TEXT),
                        (info_panel.x + 20, yy))
            yy += FONT_INFO.get_height() + 3

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if logout_btn.collidepoint(ev.pos):
                    return "__logout__"
                if cancel.collidepoint(ev.pos):
                    return "menu"
                for rect, dname in zip(btn_rects, AVAILABLE_DATASETS):
                    if rect.collidepoint(ev.pos):
                        return dname
        pygame.display.flip(); clock.tick(30)


def make_prompt_text(points):
    if points is None:
        return (f"Mission Briefing:\n"
                f"Sabotage the blue-team model!\n"
                f"Disable feature columns to drop its accuracy.\n"
                f"{ROUNDS} rounds to strike their systems — choose wisely.")
    if points == 0:
        return ("MISS…\n"
                "Their firewall barely flinched.\n"
                "Dig deeper or tap ‘?’ for recon — but your system\nwill slash 80 % of points gained.")
    if points < 60:
        return ("DIRECT HIT!\n"
                "Sensors are on alert and signs of a potential attack \nhave been discovered.\n"
                "Keep chiselling away — a breach is within reach.")
    return ("CRITICAL HIT!\n"
            "ML defenses collapse; malicious traffic floods in.\n"
            "You are ecstatic — ride the chaos for maximum damage!")


def round_ui(screen, ds, cols, max_pick, base_acc, metrics, uid,
             round_num, total_rounds, current_score, last_pts=None):
    clock = pygame.time.Clock()
    scroll = 0
    hint = False
    show_mi = False
    dragging = False
    drag_start_y = 0
    scroll_start = 0
    col_state = {c: False for c in cols}

    mi_img = None
    img_name = DATASET_TO_IMAGE.get(ds)
    if img_name and os.path.exists(img_name):
        try:
            mi_img = pygame.image.load(img_name).convert()
        except Exception as exc:
            print("[WARN] Could not load MI image:", exc)
            mi_img = None

    box = pygame.Rect(50, 150, COL_BOX_W, COL_BOX_H)
    metrics_panel = pygame.Rect(box.right + 40, 150, 330, 140)
    round_panel   = pygame.Rect(metrics_panel.right + 20, 150,  50, 60)
    score_panel   = pygame.Rect(round_panel.right  + 40, 150, 250, 60)
    panel_x = box.right + 40
    panel_y = metrics_panel.bottom + 25
    panel_w = SCREEN_WIDTH - panel_x - 60
    panel_h = SCREEN_HEIGHT - panel_y - 60
    comment_panel = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

    hint_rect  = pygame.Rect(900, 150, 60, 60)
    apply_rect = pygame.Rect(900, 220, 180, 60)

    logout_rect = pygame.Rect(0, 0, LOGOUT_W, LOGOUT_H)
    quit_rect   = pygame.Rect(SCREEN_WIDTH - 110, 80, 90, 32)

    prompt_lines = wrap_text(make_prompt_text(last_pts),
                             FONT_SMALL, comment_panel.width - 40)

    while True:
        screen.fill(COLOR_BG)
        mpos = pygame.mouse.get_pos()

        draw_uid_chip(screen, uid, logout_rect)
        draw_button_3d(screen, quit_rect, "Quit",
                       quit_rect.collidepoint(mpos), FONT_INFO)
        hdr = FONT_LARGE.render(f"Remove Columns – {ds}", True, COLOR_TEXT)
        screen.blit(hdr, (SCREEN_WIDTH//2 - hdr.get_width()//2, 60))
        screen.blit(FONT_MED.render(f"Accuracy: {base_acc:.2%}", True, COLOR_TEXT),
                    (SCREEN_WIDTH//2 - 100, 110))

        draw_panel_3d(screen, box, (210,210,210))
        draw_panel_3d(screen, metrics_panel, COLOR_CARD)
        draw_panel_3d(screen, comment_panel, COLOR_CARD)
        draw_panel_3d(screen, round_panel, COLOR_CARD)
        txt_round = FONT_SMALL.render(f"{round_num}/{total_rounds}", True, COLOR_TEXT)
        screen.blit(txt_round, (round_panel.centerx - txt_round.get_width()//2,
                                round_panel.centery - txt_round.get_height()//2))
        draw_panel_3d(screen, score_panel, COLOR_CARD)
        txt_score = FONT_SMALL.render(f"Score: {current_score}", True, COLOR_TEXT)
        screen.blit(txt_score, (score_panel.centerx - txt_score.get_width()//2,
                                score_panel.centery - txt_score.get_height()//2))

        yy = metrics_panel.y + 15
        for key in ["macro_f1", "roc_auc", "pr_auc", "tnr", "far"]:
            v = metrics.get(key)
            if v is not None:
                screen.blit(FONT_SMALL.render(f"{key}: {v:.3f}", True, COLOR_TEXT),
                            (metrics_panel.x + 15, yy))
                yy += 25

        yy = comment_panel.y + 20
        for ln in prompt_lines:
            screen.blit(FONT_SMALL.render(ln, True, COLOR_TEXT),
                        (comment_panel.x + 20, yy))
            yy += FONT_SMALL.get_height() + 2

        visible = (COL_BOX_H - 20) // 30
        shown = cols[scroll:scroll+visible]
        for i, name in enumerate(shown):
            y_pos = box.y + 20 + i*30
            cb = pygame.Rect(box.x + 10, y_pos, 20, 20)
            pygame.draw.rect(screen,
                             (100,200,100) if col_state[name] else COLOR_WHITE, cb)
            if not col_state[name]:
                pygame.draw.rect(screen, COLOR_TEXT, cb, 1)
            screen.blit(FONT_SMALL.render(str(name), True, COLOR_TEXT),
                        (cb.right + 5, y_pos-2))
        thumb = draw_scrollbar(screen, box, len(cols), visible, scroll)

        draw_button_3d(screen, hint_rect, "?", hint_rect.collidepoint(mpos), FONT_LARGE)
        draw_button_3d(screen, apply_rect, "Apply", apply_rect.collidepoint(mpos))
        if show_mi and mi_img:
            screen.blit(mi_img, (SCREEN_WIDTH - mi_img.get_width() - 20,
                                 SCREEN_HEIGHT - mi_img.get_height() - 20))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if logout_rect.collidepoint(ev.pos):
                    return "__logout__", False, last_pts
                if quit_rect.collidepoint(ev.pos):
                    return "__quit__", False, last_pts

            if ev.type == pygame.MOUSEWHEEL:
                if ev.y > 0 and scroll > 0:
                    scroll -= 1
                if ev.y < 0 and scroll < len(cols) - visible:
                    scroll += 1

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if thumb and thumb.collidepoint(ev.pos):
                    dragging = True
                    drag_start_y = ev.pos[1]
                    scroll_start = scroll
                else:
                    if box.collidepoint(ev.pos):
                        idx = (ev.pos[1] - (box.y + 20)) // 30
                        if 0 <= idx < len(shown):
                            name = shown[idx]
                            if col_state[name]:
                                col_state[name] = False
                            elif sum(col_state.values()) < max_pick:
                                col_state[name] = True
                    scroll = handle_scroll_click(box, len(cols), visible, scroll, ev)
                    if hint_rect.collidepoint(ev.pos):
                        hint = True; show_mi = not show_mi
                    if apply_rect.collidepoint(ev.pos):
                        return [c for c,v in col_state.items() if v], hint, None

            if ev.type == pygame.MOUSEMOTION and dragging and thumb:
                track_h = box.height - 20
                max_scroll = len(cols) - visible
                if track_h-thumb.height > 0 and max_scroll > 0:
                    delta = ev.pos[1] - drag_start_y
                    ratio = max_scroll / (track_h-thumb.height)
                    scroll = int(scroll_start + delta*ratio)
                    scroll = max(0, min(scroll, max_scroll))

            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False

        pygame.display.flip(); clock.tick(30)


def run_attacker_game_ui(user_id: str):
    pygame.display.set_caption("Attacker Scenario UI")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    _ensure_scoreboard_doc_exists(user_id)

    while True:
        dataset = pick_dataset_ui(screen, user_id)
        if dataset in ("menu", "__logout__", None):
            return dataset

        {
            "cicids2017": splash_cicids,
            "unsw_nb15":  splash_unsw,
            "ctu13":      splash_ctu13,
            "iot23":      splash_iot23,
            "ember":      splash_ember,
            "malware":    splash_malware,
        }[dataset](screen)

        session_score = 0
        last_points   = None

        mdl, raw_cols, _acc, _met, _sc, url = load_baseline_model(dataset)
        if raw_cols is None:
            continue
        col_list = sorted([c for c in raw_cols if str(c).lower().strip() != "label"])

        try:
            base_json = call_backend(dataset, [])
        except Exception as exc:
            print("[ERROR] backend unreachable:", exc); continue
        cur_acc, cur_met = base_json["accuracy"], base_json["metrics"]

        df = (fetch_dataset_via_url(url)
              if url else fetch_dataset_from_firebase(dataset))
        work_df = df if df is not None and "label" in df.columns else None

        quit_to_picker = False

        for rnd_idx in range(1, ROUNDS+1):
            max_pick = (
                CICIDS_PICK_LIMITS  [rnd_idx-1] if dataset=="cicids2017" else
                UNSW_PICK_LIMITS    [rnd_idx-1] if dataset=="unsw_nb15"  else
                CTU_PICK_LIMITS     [rnd_idx-1] if dataset=="ctu13"      else
                IOT23_PICK_LIMITS   [rnd_idx-1] if dataset=="iot23"      else
                EMBER_PICK_LIMITS   [rnd_idx-1] if dataset=="ember"      else
                MALWARE_PICK_LIMITS [rnd_idx-1]
            )

            selection, used_hint, _ = round_ui(
                screen, dataset, col_list, max_pick,
                cur_acc, cur_met, user_id,
                rnd_idx, ROUNDS, session_score, last_points
            )
            if selection in ("__logout__", "__quit__", None):
                if selection == "__logout__":
                    return "__logout__"
                if selection == "__quit__":
                    quit_to_picker = True
                break

            try:
                resp = call_backend(dataset, selection)
            except Exception as exc:
                print("[ERROR] backend call failed:", exc); break
            new_acc, new_met = resp["accuracy"], resp["metrics"]

            drop = cur_acc - new_acc
            pts = (
                int((drop*100)/0.10) * CICIDS_POINTS_FACTOR  if dataset=="cicids2017" else
                int((drop*100)/0.10) * UNSW_POINTS_FACTOR    if dataset=="unsw_nb15"  else
                int((drop*100)/0.10) * CTU_POINTS_FACTOR     if dataset=="ctu13"      else
                int((drop*100)/0.10) * IOT23_POINTS_FACTOR   if dataset=="iot23"      else
                int((drop*100)/0.10) * EMBER_POINTS_FACTOR   if dataset=="ember"      else
                int((drop*100)/0.10) * MALWARE_POINTS_FACTOR
            )
            if used_hint and pts:
                pts = int(pts*0.2)

            update_user_score(user_id, pts)
            if pts:
                add_attacker_score(user_id, dataset, pts)

            session_score += pts
            last_points    = pts

            col_list = sorted([c for c in col_list if c not in selection])
            if work_df is not None:
                work_df = work_df[col_list + ["label"]]
                img = regenerate_mi_chart(work_df, dataset, rnd_idx)
                if img:
                    DATASET_TO_IMAGE[dataset] = img

            cur_acc, cur_met = new_acc, new_met
            pygame.time.wait(900)

        if not quit_to_picker:
            summary_splash(screen, dataset, cur_met, session_score)


def _ensure_scoreboard_doc_exists(uid):
    ref = get_firestore_db().collection("scoreboards").document(uid)
    if not ref.get().exists:
        ref.set({"score": 0})


if __name__ == "__main__":
    os.environ.setdefault("URL")
    print("Exit:", run_attacker_game_ui("test_user_123"))
