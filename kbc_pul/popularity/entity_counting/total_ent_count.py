from typing import Optional, Dict, Set

import pandas as pd

from kbc_pul.data_structures.entity_domains import EntityDomainID
from kbc_pul.popularity.entity_count_db import EntityCountDB

EntityStr = str
RelationStr = str

# class AbstractCountCOntroller


class EntityTripleCountController:
    """
    Counts for a given entity the total number of triples it occurs in,
         over all relations (default) or the given relations
         and
         over both Subject/Object positions (default) or the given positions.

    """

    def __init__(self, entity_count_db: EntityCountDB):
        self.entity_count_db: EntityCountDB = entity_count_db

    def get_count_of(self,
                     entity: EntityStr,
                     relations_to_include: Optional[Set[EntityDomainID]] = None,
                     positions_to_include: Optional[Set[EntityDomainID]] = None
                     ) -> int:
        if positions_to_include is None:
            positions_to_include = {EntityDomainID.SUBJECT, EntityDomainID.OBJECT}
        if relations_to_include is None:
            relations_to_include = self.entity_count_db.all_relations_set

        o_relation_to_position_to_count_map: Optional[
            Dict[
                RelationStr,
                Dict[EntityDomainID, int]
            ]
        ] = self.entity_count_db.ent_to_relation_to_position_to_count_map.get(entity, None)
        if o_relation_to_position_to_count_map is None:
            return 0
        else:
            total_count: int = 0
            relation: str
            position_to_count_map: Dict[EntityDomainID, int]
            for relation, position_to_count_map in o_relation_to_position_to_count_map.items():
                if relation in relations_to_include:
                    for entity_domain, count in position_to_count_map.items():
                        if entity_domain in positions_to_include:
                            total_count += count
            return total_count

    def materialize_entity_to_count_map(self,
                                        relations_to_include: Optional[Set[EntityDomainID]] = None,
                                        positions_to_include: Optional[Set[EntityDomainID]] = None
                                        ) -> Dict[EntityStr, int]:
        if positions_to_include is None:
            positions_to_include = {EntityDomainID.SUBJECT, EntityDomainID.OBJECT}
        if relations_to_include is None:
            relations_to_include = self.entity_count_db.all_relations_set

        entity_to_count_map: Dict[EntityStr, int] = dict()
        for entity in self.entity_count_db.ent_to_relation_to_position_to_count_map.keys():
            entity_to_count_map[entity] = self.get_count_of(
                entity=entity,
                relations_to_include=relations_to_include,
                positions_to_include=positions_to_include
            )
        return entity_to_count_map

    def get_all_relations_but(self, relations_to_exclude: Set[RelationStr]) -> Set[str]:
        return self.entity_count_db.get_all_relations_but(relations_to_exclude=relations_to_exclude)

    @classmethod
    def entity_to_count_map_to_df(cls, entity_to_count_map: Dict[EntityStr, int]) -> pd.DataFrame:
        all_rows = []
        for entity, popularity in entity_to_count_map.items():
            all_rows.append([entity, popularity])
        return pd.DataFrame(
            data=all_rows,
            columns=["entity", "count"]
        )
