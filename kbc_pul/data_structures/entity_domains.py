from enum import Enum


class EntityDomainID(Enum):
    """
    Enum represesenting the positions an entity can take in a triple: Subject or Object.

    """

    SUBJECT = "Subject"
    OBJECT = "Object"

    def get_other(self) -> 'EntityDomainID':
        if self is EntityDomainID.SUBJECT:
            return EntityDomainID.OBJECT
        elif self is EntityDomainID.OBJECT:
            return EntityDomainID.SUBJECT
        else:
            raise Exception(f"{self} is not recognized as Subject or Object")
