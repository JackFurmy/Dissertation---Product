

import sys
import random
import pygame

from user_session import set_user_model, get_user_score, update_user_score
from ml_integration import load_baseline_model


def resource_path(rel_path: str) -> str:                    
    base = getattr(sys, "_MEIPASS", __file__ if "__file__" in globals() else ".")
    base = base if isinstance(base, str) else base.decode()  
    return  os.path.join(os.path.dirname(base), rel_path)

AVAILABLE_BASELINES = [
    "cicids2017",
    "unsw_nb15",
    "iot23",
    "ember",
    "ctu13",
    "malware_detection",
    "iot_network",
]

MAX_COLUMNS_REMOVABLE = 5


def run_attacker_scenario(user_id: str) -> None:
    pygame.init()
    screen = pygame.display.set_mode((1000, 700))
    pygame.display.set_caption("Attacker Scenario – Remove Leaky Columns")

    dataset_name = show_dataset_picker(screen, AVAILABLE_BASELINES)
    if not dataset_name:
        print("[INFO] User cancelled – exiting attacker scenario.")
        pygame.quit()
        return

    model_obj, col_list, baseline_acc = load_baseline_model(dataset_name)
    if model_obj is None:
        print(f"[ERROR] Could not load baseline ⇒ {dataset_name}")
        pygame.quit()
        return
    print(f"[INFO] Baseline loaded ⇒ {dataset_name}, baseline_acc={baseline_acc:.4f}")

    set_user_model(user_id, model_obj, "attacker")

    removed_columns: set[str] = set()
    user_score      = get_user_score(user_id) or 0

    chosen_cols = show_column_removal_ui(screen, col_list, removed_columns, dataset_name)
    if chosen_cols is None:
        print("[INFO] User cancelled – ending attacker scenario.")
        pygame.quit()
        return

    removed_columns.update(chosen_cols)
    if len(removed_columns) > MAX_COLUMNS_REMOVABLE:
        removed_columns = set(list(removed_columns)[:MAX_COLUMNS_REMOVABLE])

    new_acc      = approximate_new_accuracy(baseline_acc, removed_columns)
    drop_amount  = baseline_acc - new_acc
    points_earned = 50 if drop_amount > 0.10 else 20 if drop_amount > 0.05 else 0

    if points_earned:
        update_user_score(user_id, points_earned)
        user_score += points_earned

    show_final_attacker_results(
        screen, user_score, removed_columns, baseline_acc, new_acc
    )
    pygame.quit()


def show_dataset_picker(screen, dataset_list):
    if not dataset_list:
        return None
    return dataset_list[0]


def show_column_removal_ui(screen, all_cols, removed_set, dataset_name):
    picks = [c for c in all_cols if c not in removed_set][:2]
    return picks if picks else None

def approximate_new_accuracy(baseline_acc, removed_cols):
    degrade = random.uniform(0.0, 0.12)
    new_acc = baseline_acc - degrade
    return max(new_acc, 0.0)


def show_final_attacker_results(
    screen, user_score, removed_cols, base_acc, new_acc
):
    print("\n=== Attacker scenario summary ===")
    print(f"Removed columns   : {sorted(removed_cols)}")
    print(f"Baseline accuracy : {base_acc:.4f}")
    print(f"New accuracy      : {new_acc:.4f}")
    print(f"Updated scoreboard: {user_score}")


if __name__ == "__main__":
    test_user_id = "attacker_uid_001"
    run_attacker_scenario(test_user_id)
