"""Simple data models for a pet store."""

from dataclasses import dataclass


@dataclass
class Pet:
    name: str
    species: str
    age: int
    owner: "Owner | None" = None

    def greet(self) -> str:
        return f"Hi, I'm {self.name} the {self.species}!"

    def is_senior(self) -> bool:
        return self.age > 10


@dataclass
class Owner:
    name: str
    email: str

    def adopt(self, pet: Pet) -> None:
        pet.owner = self

    def summary(self) -> str:
        return f"{self.name} ({self.email})"
