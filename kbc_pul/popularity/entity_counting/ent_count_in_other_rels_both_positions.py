from typing import Optional, Dict

import pandas as pd

from kbc_pul.popularity.entity_counting.total_ent_count import EntityTripleCountController

EntityStr = str


class EntityCountInOtherRelationsBothPositions:

    def __init__(self,
                 target_relation: str,
                 entity_triple_count_controller: EntityTripleCountController,
                 verbose: bool = False
                 ):
        self.target_relation: str = target_relation
        self.verbose: bool = verbose
        self.entity_triple_count_controller: EntityTripleCountController = entity_triple_count_controller

        self.o_entity_to_count_map: Optional[
            Dict[
                EntityStr,
                int
            ]
        ] = None

        self.relations_to_include = self.entity_triple_count_controller.get_all_relations_but(
            {target_relation}
        )
        self.positions_to_include = None

    def get_count(self, entity: EntityStr) -> int:
        if self.o_entity_to_count_map is not None:
            return self.o_entity_to_count_map.get(entity, 0)
        else:
            return self.entity_triple_count_controller.get_count_of(
                entity=entity,
                relations_to_include=self.relations_to_include,
                positions_to_include=self.positions_to_include
            )

    def are_counts_materialized(self) -> bool:
        return self.o_entity_to_count_map is not None

    def materialize_entity_counts(self) -> None:
        if self.verbose:
            print("Materializing entity counts...")
        self.o_entity_to_count_map = self.entity_triple_count_controller.materialize_entity_to_count_map(
            relations_to_include=self.relations_to_include,
            positions_to_include=self.positions_to_include
        )

    def get_entity_counts_as_df(self) -> pd.DataFrame:
        if not self.are_counts_materialized():
            self.materialize_entity_counts()
        return EntityTripleCountController.entity_to_count_map_to_df(self.o_entity_to_count_map)
