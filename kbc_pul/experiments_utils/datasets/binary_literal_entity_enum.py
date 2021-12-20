from enum import Enum


class BinaryLiteralEntity(Enum):
    SUBJECT = "subject"
    OBJECT = "object"

    def get_column_name(self) -> str:
        if self.value == "subject":
            return "Subject"
        elif self.value == "object":
            return "Object"
        else:
            raise Exception(f"Cannot find valid column name for BinaryLiteralEntity with value {self.value}")

    def get_other_position(self) -> 'BinaryLiteralEntity':
        if self.value == "subject":
            return BinaryLiteralEntity.OBJECT
        elif self.value == "object":
            return BinaryLiteralEntity.SUBJECT
        else:
            raise Exception(f"Cannot return other position for BinaryLiteralEntity with value {self.value}")
