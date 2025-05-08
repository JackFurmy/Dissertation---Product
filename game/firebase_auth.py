
import os
import sys
import certifi                          
import pyrebase

from firebase_init import get_firestore_db
import user_session

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel)

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

def init_pyrebase():

    config = {
        "apiKey":            "KEY",
        "authDomain":        "DOMAIN",
        "projectId":         "ID",
        "storageBucket":     "BUCKET",
        "messagingSenderId": "ID",
        "appId":             "ID",
        "measurementId":     "ID",
        "databaseURL":       "",                       
    }
    firebase = pyrebase.initialize_app(config)
    print("[INFO] Pyrebase initialised for 'mldiss'.")
    return firebase

def create_fresh_model_for_new_user(local_id: str):
    user_session.create_fresh_model_for_new_user(local_id)

def sign_up_user(firebase, email, password, display_name=None):
    db   = get_firestore_db()
    auth = firebase.auth()
    try:
        user      = auth.create_user_with_email_and_password(email, password)
        local_id  = user.get("localId")
        print(f"[INFO] Sign-up success ⇒ localId={local_id}")

        if display_name:
            store_additional_profile_data(db, user, {"displayName": display_name})

        user_session.create_fresh_model_for_new_user(local_id)
        return user
    except Exception as e:
        print(f"[WARN] sign_up_user ⇒ {e}")
        return None


def login_user(firebase, email, password):
    auth = firebase.auth()
    try:
        user     = auth.sign_in_with_email_and_password(email, password)
        local_id = user.get("localId")
        print(f"[INFO] Login success ⇒ localId={local_id}")
        return user
    except Exception as e:
        print(f"[WARN] login_user ⇒ {e}")
        return None

def store_additional_profile_data(db, user_dict, profile_dict):
    local_id = user_dict.get("localId")
    if not local_id:
        return
    try:
        db.collection("users").document(local_id).set(profile_dict, merge=True)
        print("[INFO] Stored profile ⇒", profile_dict)
    except Exception as e:
        print(f"[WARN] store_additional_profile_data ⇒ {e}")


def refresh_user_token(firebase, user_dict):
    auth = firebase.auth()
    try:
        new_data                 = auth.refresh(user_dict["refreshToken"])
        user_dict["idToken"]     = new_data["idToken"]
        user_dict["refreshToken"] = new_data["refreshToken"]
        print("[INFO] Token refreshed.")
        return user_dict
    except Exception as e:
        print(f"[WARN] refresh_user_token ⇒ {e}")
        return None

def login_or_signup_cli():
    firebase = init_pyrebase()
    choice   = input("Type 'login' or 'signup': ").strip().lower()
    email    = input("Email: ")
    pwd      = input("Password: ")

    if choice == "signup":
        display_name = input("Display name? ")
        return sign_up_user(firebase, email, pwd, display_name)
    elif choice == "login":
        return login_user(firebase, email, pwd)
    else:
        print("[WARN] Unknown choice ⇒ returning None.")
        return None

if __name__ == "__main__":
    user = login_or_signup_cli()
    if user:
        print("User ⇒", user)
    else:
        print("No user ⇒ done.")
