import abc


class Base64Convertible(abc.ABC):
    @abc.abstractmethod
    def get_base64(self) -> str: ...
