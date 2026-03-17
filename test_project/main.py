"""Entry point for the pet store."""

from models import Pet, Owner
from store import add_pet, adopt_pet, list_seniors, store_summary


def main() -> None:
    # Add some pets
    buddy = add_pet("Buddy", "dog", 3)
    whiskers = add_pet("Whiskers", "cat", 12)
    goldie = add_pet("Goldie", "fish", 1)

    # Create owner
    alice = Owner(name="Alice", email="alice@example.com")

    # Adopt
    adopt_pet(alice, "Buddy")

    # Report
    seniors = list_seniors()
    for s in seniors:
        print(s.greet())

    print(store_summary())


if __name__ == "__main__":
    main()
