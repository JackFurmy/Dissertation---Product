

import os
import sys
import textwrap
import pygame

pygame.init()

def resource_path(rel: str) -> str:
 
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel)

SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Game Menu – Defender vs Attacker")

FONT_TITLE  = pygame.font.SysFont(None, 54)
FONT_BODY   = pygame.font.SysFont(None, 26)
FONT_BUTTON = pygame.font.SysFont(None, 38)
FONT_UID    = pygame.font.SysFont(None, 22)

COLOR_BG           = (247, 255, 246)
COLOR_TEXT         = (0, 0, 0)
COLOR_PANEL        = (255, 255, 255)
COLOR_SHADOW       = (0, 0, 0, 60)
COLOR_BUTTON       = (96, 109, 128)
COLOR_BUTTON_HOVER = (118, 131, 148)
COLOR_UID_CHIP     = (240, 248, 235)
COLOR_UID_OUTLINE  = (200, 208, 195)
COLOR_WHITE        = (255, 255, 255)

PANEL_W, PANEL_H   = 1000, 620
PANEL_RADIUS       = 16
SHADOW_OFFSET      = 8

BUTTON_W, BUTTON_H = 280, 85
BUTTON_RADIUS      = 28
BUTTON_GAP         = 120        

LB_W, LB_H         = 300, 70     

UID_MARGIN_X, UID_MARGIN_Y = 20, 14
LOGOUT_W, LOGOUT_H         = 100, 36
LOGOUT_RADIUS              = 14

WRAP_CHARS   = 80
LINE_SPACING = 6
PARA_SPACING = 20

def center_wrap(text: str):
    lines = textwrap.wrap(text.strip(), WRAP_CHARS)
    return [FONT_BODY.render(ln, True, COLOR_TEXT) for ln in lines]

def draw_button(rect: pygame.Rect, label: str, hover=False):
    pygame.draw.rect(
        screen,
        COLOR_BUTTON_HOVER if hover else COLOR_BUTTON,
        rect,
        border_radius=BUTTON_RADIUS,
    )
    surf = FONT_BUTTON.render(label, True, COLOR_WHITE)
    screen.blit(surf, (rect.centerx - surf.get_width() // 2,
                       rect.centery - surf.get_height() // 2))

def draw_chip(rect: pygame.Rect, label: str):
    pygame.draw.rect(screen, COLOR_UID_CHIP, rect, border_radius=LOGOUT_RADIUS)
    pygame.draw.rect(screen, COLOR_UID_OUTLINE, rect, 1, border_radius=LOGOUT_RADIUS)
    surf = FONT_UID.render(label, True, COLOR_TEXT)
    screen.blit(surf, (rect.centerx - surf.get_width() // 2,
                       rect.centery - surf.get_height() // 2))

def show_main_menu(uid: str) -> str:
    clock = pygame.time.Clock()

    attacker_p = (
        "Attacker Game (baseline models): step into the shoes of an adversary. "
        "Strip key features from a defensive ML model. "
        "Drop its recall enough and malicious traffic walks right in—points for you!"
    )
    defender_p = (
        "Defender Game (incremental models): you're the blue-team hero. "
        "Feed fresh chunks to a tiny model, boosting recall while holding FAR low. "
        "Mentor your model to greatness and rack up points."
    )
    role_p = (
        "Pick your role and learn how an ML model’s fate hinges on what data flows "
        "through—or disappears from—its pipeline."
    )

    attacker_lines  = center_wrap(attacker_p)
    defender_lines  = center_wrap(defender_p)
    role_lines      = center_wrap(role_p)
    blocks          = [attacker_lines, defender_lines, role_lines]

    panel  = pygame.Rect(0, 0, PANEL_W, PANEL_H)
    panel.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    shadow = panel.move(SHADOW_OFFSET, SHADOW_OFFSET)

    uid_surf   = FONT_UID.render(f"UID: {uid}", True, COLOR_TEXT)
    chip_w     = uid_surf.get_width() + LOGOUT_W + UID_MARGIN_X * 3
    chip_rect  = pygame.Rect(SCREEN_WIDTH - chip_w - 20, 20, chip_w, LOGOUT_H)
    uid_pos    = (chip_rect.x + UID_MARGIN_X,
                  chip_rect.y + (LOGOUT_H - uid_surf.get_height()) // 2)
    logout_rect = pygame.Rect(
        chip_rect.right - LOGOUT_W - UID_MARGIN_X,
        chip_rect.y, LOGOUT_W, LOGOUT_H)

    btn_y = panel.bottom - 110
    defender_rect = pygame.Rect(
        panel.centerx - BUTTON_W - BUTTON_GAP//2, btn_y, BUTTON_W, BUTTON_H)
    attacker_rect = defender_rect.move(BUTTON_W + BUTTON_GAP, 0)

    y_cursor = panel.y + 40 + FONT_TITLE.get_height() + 30
    block_tops = []
    for bl in blocks:
        block_tops.append(y_cursor)
        y_cursor += sum(s.get_height()+LINE_SPACING for s in bl) + PARA_SPACING
    leaderboard_rect = pygame.Rect(panel.centerx - LB_W//2, y_cursor, LB_W, LB_H)

    while True:
        mx, my = pygame.mouse.get_pos()
        screen.fill(COLOR_BG)

        sh_surf = pygame.Surface(panel.size, pygame.SRCALPHA)
        sh_surf.fill(COLOR_SHADOW)
        screen.blit(sh_surf, shadow.topleft)
        pygame.draw.rect(screen, COLOR_PANEL, panel, border_radius=PANEL_RADIUS)

        title_surf = FONT_TITLE.render("Welcome to 'Defender & Attacker'", True, COLOR_TEXT)
        screen.blit(title_surf, (panel.centerx - title_surf.get_width()//2, panel.y + 40))

        for bl, top in zip(blocks, block_tops):
            y = top
            for line in bl:
                screen.blit(line, (panel.centerx - line.get_width()//2, y))
                y += line.get_height() + LINE_SPACING

        draw_button(defender_rect,   "Defender",   defender_rect.collidepoint(mx, my))
        draw_button(attacker_rect,   "Attacker",   attacker_rect.collidepoint(mx, my))
        draw_button(leaderboard_rect, "Leaderboard", leaderboard_rect.collidepoint(mx, my))

        draw_chip(chip_rect := chip_rect, label="")       
        screen.blit(uid_surf, uid_pos)
        draw_chip(logout_rect, "Log Out")

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if defender_rect.collidepoint(ev.pos):   return "defender"
                if attacker_rect.collidepoint(ev.pos):   return "attacker"
                if leaderboard_rect.collidepoint(ev.pos):return "leaderboard"
                if logout_rect.collidepoint(ev.pos):     return "logout"

        pygame.display.flip(); clock.tick(30)

def run_menu(user_id: str) -> str:
    return show_main_menu(user_id)

if __name__ == "__main__":
    print(run_menu("MY_UID_123"))
