import string
from string import printable
from typing import Optional, Dict

import unidecode

space_substitution: str = '_'


class TokenEncoder:
    """
    Prolog engines do not always agree with a lot of unicode characters.
    This class is used to clean up these characters.

    """
    def __init__(self, should_cache: bool):
        self.should_cache: bool = should_cache
        self.encoding_map: Optional[Dict[str, str]] = None
        if should_cache:
            self.encoding_map: Dict[str, str] = dict()

    def encode(self, entity_or_relation: str, is_entity: bool = False) -> str:
        er_atom: str
        if self.should_cache:
            er_atom: str = self.encoding_map.get(entity_or_relation, None)
            if er_atom is None:
                er_atom: str = self.quick_and_dirty_clean_as_atom(entity_or_relation, is_entity=is_entity)
                self.encoding_map[entity_or_relation] = er_atom
        else:
            er_atom = self.quick_and_dirty_clean_as_atom(entity_or_relation, is_entity=is_entity)
        return er_atom

    @staticmethod
    def quick_and_dirty_clean_as_atom(entity_or_relation: str, is_entity: bool = False) -> str:
        """
        Fom Learn Prolog Now (http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse2):

        An atom is either:

            1. A string of characters made up of
                * upper-case letters,
                * lower-case letters,
                * digits, and
                * the underscore character,
              that begins with a lower-case letter.
              Here are some examples:
                butch , big_kahuna_burger , listens2Music and playsAirGuitar.
            2. An arbitrary sequence of characters enclosed in single quotes.
               For example ’ Vincent ’, ’ The  Gimp ’, ’ Five_Dollar_Shake ’, ’ &^%&#@$  &* ’, and ’   ’.
               The sequence of characters between the single quotes is called the atom name.
               Note that we are allowed to use spaces in such atoms;
               in fact, a common reason for using single quotes is so we can do precisely that.

            3. A string of special characters.
               Here are some examples:
                    @= and ====> and ; and :- are all atoms.
                As we have seen, some of these atoms, such as ; and :- have a pre-defined meaning.


        Here, we DIRTILY convert everything to the first type of atom.
        Note,
            THIS EXCLUDES INTEGERS

        :param entity_or_relation:
        :return:
        """
        if is_entity:
            entity_str = unidecode.unidecode(entity_or_relation)
            entity_str = entity_str.replace("/", "_")
            entity_str = TokenEncoder.remove_non_ascii_characters(entity_str)
            entity_str = entity_str.replace("\\", "")
            entity_str = entity_str.replace(" ", "")
            entity_str = entity_str.replace('&amp;', '')
            if entity_str[0] not in string.ascii_lowercase:
                entity_str = 'e' + entity_str
            entity_str = "'" + entity_str.replace("'", "") + "'"
            return entity_str

        else:
            entity_str: str = unidecode.unidecode(entity_or_relation)
            entity_str = entity_str.replace("/", "_")
            entity_str = entity_str.replace("\\", "")
            # first letter must be lower case
            if not entity_or_relation[0].islower():
                entity_str = entity_str[0].lower() + entity_str[1:]

            entity_str = ''.join(TokenEncoder._transform_char(i) for i in entity_str)
            try:
                if entity_str[0].isdigit():
                    entity_str = 'n' + entity_str
                elif entity_str[0] not in string.ascii_lowercase:
                    entity_str = 'r' + entity_str

            except IndexError as err:
                print(f"Entity not valid: '{entity_or_relation}' became '{entity_str}'")
                raise err

            entity_str = TokenEncoder.remove_non_ascii_characters(entity_str)

            return entity_str

    @staticmethod
    def remove_non_ascii_characters(token: str) -> str:
        return ''.join(filter(lambda x: x in printable, token))

    @staticmethod
    def _transform_char(char: str) -> str:
        if char.isnumeric():
            return char
        elif char.isalpha():
            if char.isupper():
                return char.lower()
            else:
                return char
        elif char == "_":
            return "_"
        elif char == ' ':
            return space_substitution
        else:
            return ''
