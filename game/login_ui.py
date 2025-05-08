
import pygame
import sys
import textwrap

from menu_ui import run_menu
from firebase_auth import init_pyrebase, login_user, sign_up_user
import user_session  

pygame.init()


# ──────── Global styling constants ────────
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("A Series Game On ML - Auth Flow with Info Page")

TITLE_FONT_SIZE = 60
LABEL_FONT_SIZE = 36
INFO_FONT_SIZE  = 28

font_title = pygame.font.SysFont(None, TITLE_FONT_SIZE)
font_label = pygame.font.SysFont(None, LABEL_FONT_SIZE)
font_info  = pygame.font.SysFont(None, INFO_FONT_SIZE)

COLOR_BG           = (247, 255, 246)   
COLOR_TEXT         = (0, 0, 0)
COLOR_INPUT_BG     = (188, 235, 203)
COLOR_BUTTON       = (132, 145, 163)
COLOR_BUTTON_HOVER = (150, 163, 180)

email_text, password_text = "", ""
active_field = 0        


def show_information_page() -> None:
    CARD_W, CARD_H = 900, 560
    CARD_COLOR     = (255, 255, 255)
    CARD_SHADOW    = (200, 210, 200, 150)
    SHADOW_OFFSET  = 6

    font_heading = pygame.font.SysFont(None, 34)
    font_body    = pygame.font.SysFont(None, 26)

    body_lines = [
        "This application is part of a dissertation on gamified cybersecurity training.",
        "You will play interactive, ML-focused scenarios as a 'defender' or an 'attacker'.",
        "",
        "Participation is voluntary — you can leave anytime with no penalty.",
        "Email / password are stored securely; no personal data appears in-game.",
        "The application complies with standard data-protection regulations (GDPR).",
        "",
        "By clicking <Acknowledge> you confirm you understand these conditions.",
        "Need your data removed?",
        "",
        "",
        "Contact:  21312564@stu.mmu.ac.uk",
    ]

    wrapped_surfs = []
    for raw in body_lines:
        if raw == "":
            wrapped_surfs.append(None)
            continue
        for part in textwrap.wrap(raw, width=80):
            wrapped_surfs.append(font_body.render(part, True, COLOR_TEXT))

    ack_rect = pygame.Rect(0, 0, 300, 60)
    ack_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + CARD_H // 2 - 50)

    clock = pygame.time.Clock()
    while True:
        screen.fill(COLOR_BG)
        shadow_rect = pygame.Rect(
            SCREEN_WIDTH // 2 - CARD_W // 2 + SHADOW_OFFSET,
            SCREEN_HEIGHT // 2 - CARD_H // 2 + SHADOW_OFFSET,
            CARD_W, CARD_H,
        )
        shadow = pygame.Surface((CARD_W, CARD_H), pygame.SRCALPHA)
        shadow.fill(CARD_SHADOW)
        screen.blit(shadow, shadow_rect.topleft)

        card_rect = pygame.Rect(
            SCREEN_WIDTH // 2 - CARD_W // 2,
            SCREEN_HEIGHT // 2 - CARD_H // 2,
            CARD_W, CARD_H,
        )
        pygame.draw.rect(screen, CARD_COLOR, card_rect, border_radius=12)

        hsurf = font_heading.render("Information / Consent", True, COLOR_TEXT)
        screen.blit(
            hsurf,
            (
                card_rect.centerx - hsurf.get_width() // 2,
                card_rect.y + 28,
            ),
        )

        y = card_rect.y + 28 + hsurf.get_height() + 30
        for surf in wrapped_surfs:
            if surf is None:
                y += font_body.get_height()
                continue
            screen.blit(surf, (card_rect.x + 40, y))
            y += surf.get_height() + 4

        hover = ack_rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(
            screen,
            (120, 133, 150) if hover else (132, 145, 163),
            ack_rect,
            border_radius=30,
        )
        atxt = font_label.render("Acknowledge", True, (255, 255, 255))
        screen.blit(
            atxt,
            (
                ack_rect.centerx - atxt.get_width() // 2,
                ack_rect.centery - atxt.get_height() // 2,
            ),
        )

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if ack_rect.collidepoint(e.pos):
                    return

        pygame.display.flip()
        clock.tick(30)



CARD_W, CARD_H = 680, 480
CARD_COLOR     = (255, 255, 255)
CARD_SHADOW    = (0, 0, 0, 50)
CARD_RADIUS    = 14
SHADOW_OFFSET  = 8
INPUT_HEIGHT = 60
PLACE_COLOR  = (130, 130, 130)
INPUT_BORDER = (160, 200, 170)
INPUT_BORDER_F = (90, 160, 120)
BUTTON_RADIUS = 24


def draw_shadowed_card() -> pygame.Rect:
    x = SCREEN_WIDTH // 2 - CARD_W // 2
    y = SCREEN_HEIGHT // 2 - CARD_H // 2

    sh_rect = pygame.Rect(x + SHADOW_OFFSET, y + SHADOW_OFFSET, CARD_W, CARD_H)
    sh_surf = pygame.Surface((CARD_W, CARD_H), pygame.SRCALPHA)
    sh_surf.fill(CARD_SHADOW)
    screen.blit(sh_surf, sh_rect.topleft)

    bg_rect = pygame.Rect(x, y, CARD_W, CARD_H)
    pygame.draw.rect(screen, CARD_COLOR, bg_rect, border_radius=CARD_RADIUS)
    return bg_rect


def draw_text_input(rect, text, placeholder, active=False, password=False):
    border_col = INPUT_BORDER_F if active else INPUT_BORDER
    pygame.draw.rect(screen, border_col, rect, border_radius=6)
    inner = rect.inflate(-4, -4)
    pygame.draw.rect(screen, COLOR_INPUT_BG, inner, border_radius=6)

    if not text and not active:
        surf = font_label.render(placeholder, True, PLACE_COLOR)
    else:
        disp = "*" * len(text) if password else text
        surf = font_label.render(disp, True, COLOR_TEXT)

    screen.blit(
        surf,
        (inner.x + 14, inner.y + (inner.height - surf.get_height()) // 2),
    )


def draw_pill_button(rect, label, hover=False):
    color = COLOR_BUTTON_HOVER if hover else COLOR_BUTTON
    pygame.draw.rect(screen, color, rect, border_radius=BUTTON_RADIUS)
    tsurf = font_label.render(label, True, (255, 255, 255))
    screen.blit(
        tsurf,
        (rect.centerx - tsurf.get_width() // 2,
         rect.centery - tsurf.get_height() // 2),
    )

def do_login(firebase, email, pwd):
    if not email or not pwd:
        print("[INFO] Must enter email & password.")
        return None
    user = login_user(firebase, email, pwd)
    return user.get("localId") if user else None


def do_signup(firebase, email, pwd):
    if not email or not pwd:
        print("[INFO] Must enter email & password.")
        return None
    user = sign_up_user(firebase, email, pwd)
    return user.get("localId") if user else None


def handle_post_login(uid, attacker_game_ui, defender_game_ui):
    from leaderboard import run_leaderboard_ui      

    choice = run_menu(uid)
    if choice == "logout":
        return

    if choice == "attacker":
        res = attacker_game_ui.run_attacker_game_ui(uid)
        if res == "menu":
            handle_post_login(uid, attacker_game_ui, defender_game_ui)

    elif choice == "defender":
        res = defender_game_ui.run_defender_ui(uid)
        if res == "menu":
            handle_post_login(uid, attacker_game_ui, defender_game_ui)

    elif choice == "leaderboard":                   
        res = run_leaderboard_ui(uid)                
        if res == "menu":                            
            handle_post_login(uid, attacker_game_ui, defender_game_ui)
            
def main_loop(firebase):
    import attacker_game_ui, defender_game_ui

    global email_text, password_text, active_field

    clock = pygame.time.Clock()
    while True:
        card_rect = pygame.Rect(0, 0, CARD_W, CARD_H)
        card_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        email_rect    = pygame.Rect(card_rect.x + 60, card_rect.y + 120, CARD_W - 120, INPUT_HEIGHT)
        password_rect = email_rect.move(0, 100)

        login_rect  = pygame.Rect(card_rect.x + 60, card_rect.bottom - 120,
                                  (CARD_W - 160) // 2, 70)
        signup_rect = login_rect.move(login_rect.width + 40, 0)

        while True:
            screen.fill(COLOR_BG)
            draw_shadowed_card()

            title = "Defender & Attacker – A Series Game on ML"
            tsurf = font_title.render(title, True, COLOR_TEXT)
            screen.blit(tsurf, (SCREEN_WIDTH//2 - tsurf.get_width()//2, card_rect.y - 50))

            lbl_font = pygame.font.SysFont(None, 26)
            screen.blit(lbl_font.render("Email", True, COLOR_TEXT),
                        (email_rect.x, email_rect.y - 28))
            screen.blit(lbl_font.render("Password", True, COLOR_TEXT),
                        (password_rect.x, password_rect.y - 28))

            draw_text_input(email_rect, email_text, "Email", active_field == 1)
            draw_text_input(password_rect, password_text, "Password", active_field == 2, password=True)

            mx, my = pygame.mouse.get_pos()
            draw_pill_button(login_rect,  "Login",   login_rect.collidepoint(mx, my))
            draw_pill_button(signup_rect, "Sign Up", signup_rect.collidepoint(mx, my))

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if email_rect.collidepoint(e.pos):
                        active_field = 1
                    elif password_rect.collidepoint(e.pos):
                        active_field = 2
                    else:
                        active_field = 0

                    if login_rect.collidepoint(e.pos):
                        uid = do_login(firebase, email_text, password_text)
                        if uid:
                            handle_post_login(uid, attacker_game_ui, defender_game_ui)
                            email_text = password_text = ""; active_field = 0

                    elif signup_rect.collidepoint(e.pos):
                        uid = do_signup(firebase, email_text, password_text)
                        if uid:
                            handle_post_login(uid, attacker_game_ui, defender_game_ui)
                            email_text = password_text = ""; active_field = 0

                elif e.type == pygame.KEYDOWN:
                    if active_field == 1:
                        if e.key == pygame.K_BACKSPACE:
                            email_text = email_text[:-1]
                        elif e.key != pygame.K_RETURN:
                            email_text += e.unicode
                    elif active_field == 2:
                        if e.key == pygame.K_BACKSPACE:
                            password_text = password_text[:-1]
                        elif e.key != pygame.K_RETURN:
                            password_text += e.unicode

            pygame.display.flip()
            clock.tick(30)


def run_app():
    show_information_page()               
    firebase = init_pyrebase()
    main_loop(firebase)


if __name__ == "__main__":
    run_app()

