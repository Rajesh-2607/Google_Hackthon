"""
Upload Helper — Google Drive Folder ID Extractor
=================================================
Run this script to extract folder IDs from your Google Drive share links.
You only need 2 IDs: one for processed_data/ and one for results/.

Usage:
    python upload_helper.py
"""


def extract_folder_id(share_link: str) -> str:
    """Extract folder ID from a Google Drive folder link."""
    # Format: https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing
    if "/folders/" in share_link:
        return share_link.split("/folders/")[1].split("?")[0].split("/")[0]
    # Assume it's already a folder ID
    return share_link.strip()


FOLDERS = {
    "processed_data": "GDRIVE_PROCESSED_DATA_FOLDER",
    "results":        "GDRIVE_RESULTS_FOLDER",
}


def main():
    print("=" * 60)
    print("  Google Drive Folder ID Extractor")
    print("=" * 60)
    print()
    print("For each folder, paste the Google Drive share link.")
    print("(Right-click folder → Share → Copy link)")
    print()

    secrets = {}

    for folder_name, secret_key in FOLDERS.items():
        link = input(f"  {folder_name}/ share link: ").strip()
        if link:
            folder_id = extract_folder_id(link)
            secrets[secret_key] = folder_id
            print(f"  ✓ {secret_key} = {folder_id}\n")
        else:
            print(f"  ⏭ Skipped\n")

    print("\n" + "=" * 60)
    print("  PASTE THIS INTO STREAMLIT CLOUD SECRETS")
    print("=" * 60 + "\n")

    for key, value in secrets.items():
        print(f'{key} = "{value}"')

    print()
    print("=" * 60)
    print("  Done! Go to your Streamlit Cloud app →")
    print("  Settings → Secrets → paste the above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
