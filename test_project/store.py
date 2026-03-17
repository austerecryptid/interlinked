"""Pet store service layer."""

from models import Pet, Owner


inventory: list[Pet] = []


def add_pet(name: str, species: str, age: int) -> Pet:
    pet = Pet(name=name, species=species, age=age)
    inventory.append(pet)
    return pet


def find_pet(name: str) -> Pet | None:
    for pet in inventory:
        if pet.name == name:
            return pet
    return None


def adopt_pet(owner: Owner, pet_name: str) -> bool:
    pet = find_pet(pet_name)
    if pet is None:
        return False
    owner.adopt(pet)
    return True


def list_seniors() -> list[Pet]:
    return [p for p in inventory if p.is_senior()]


def store_summary() -> str:
    total = len(inventory)
    adopted = sum(1 for p in inventory if p.owner is not None)
    return f"Store: {total} pets, {adopted} adopted"
