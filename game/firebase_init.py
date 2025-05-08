
import os
import sys
import firebase_admin
from firebase_admin import credentials, firestore


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel)


_db = None


def init_firebase_admin():
    json_key_path = resource_path("json")

    if not os.path.exists(json_key_path):
        print(f"[WARN] Service-account file not found ⇒ {json_key_path}")
        return None

    try:
        cred = credentials.Certificate(json_key_path)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        print("[INFO] firebase_admin initialised (firebase_init).")
        return db

    except Exception as e:
        print(f"[ERROR] init_firebase_admin ⇒ {e}")
        return None

def get_firestore_db():
    global _db
    if _db is None:
        _db = init_firebase_admin()
    return _db

if __name__ == "__main__":
    db = get_firestore_db()
    if db:
        print("[INFO]Firestore client from firebase_init!")
