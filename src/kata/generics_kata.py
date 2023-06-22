from typing import List, TypeVar, Generic, Dict

T = TypeVar("T", str, int)


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._store: Dict[str, T] = {}

    def set_item(self, k: str, v: T) -> None:
        self._store[k] = v

    def get_item(self, k: str) -> T:
        return self._store[k]


def first(container: List[T]) -> T:
    return container[0]


def second(container: List[T]) -> T:
    return container[1]


if __name__ == "__main__":
    list_one: List[str] = ["a", "b", "c"]
    list_two: List[int] = [1, 2, 3]
    print(first(list_one))
    print(second(list_two))

    player_position_reg = Registry[str]()
    player_shirt_numb_reg = Registry[int]()
    player_position_reg.set_item('robert lewandowski', 'attack')
    player_shirt_numb_reg.set_item('robert lewandowski', 9)
    print(player_position_reg.get_item('robert lewandowski'))
    print(player_shirt_numb_reg.get_item('robert lewandowski'))
