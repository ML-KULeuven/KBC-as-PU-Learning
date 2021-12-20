from typing import Dict, Optional, Set, Tuple

import pandas as pd

from kbc_pul.data_structures.entity_domains import EntityDomainID
from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper

RelationStr = str
EntityStr = str


class EntityCountDB:
    """
    Counts how many times each entity occurs for each relation in the Subject/Object position.

    Map: Entity -> Relation -> Subject/Object position -> triple count

    """

    def __init__(self, pandas_kb_wrapper: PandasKnowledgeBaseWrapper):

        self.all_relations_set: Set[RelationStr] = set()
        self.ent_to_relation_to_position_to_count_map: Dict[
            EntityStr,
            Dict[
                RelationStr,
                Dict[EntityDomainID, int]
                ]
        ] = dict()

        self.relation_to_position_to_max_count_map: Dict[
            str,
            Dict[
                EntityDomainID,
                Tuple[EntityStr, int]
            ]
        ] = dict()
        relation_name: str
        relation_df: pd.DataFrame
        for relation_name, relation_df in pandas_kb_wrapper.map_of_predicate_to_df_entity_tuples.items():
            self.all_relations_set.add(relation_name)

            print(f"{relation_name} counting Subject")
            subj_ent_to_triple_count_map = self.count_n_triples_per_ent_in_rel_in_position(
                relation_df=relation_df,
                entity_domain=EntityDomainID.SUBJECT,
                o_ent_to_triple_count_map=dict()
            )

            max_count_for_subj: int = 0
            max_subj_ent = None
            for ent, count in subj_ent_to_triple_count_map.items():
                o_relation_to_pos_to_count_map: Optional[
                    Dict[
                        RelationStr,
                        Dict[EntityDomainID, int]
                    ]
                ] = self.ent_to_relation_to_position_to_count_map.get(ent, None)
                if o_relation_to_pos_to_count_map is None:
                    o_relation_to_pos_to_count_map = dict()
                    self.ent_to_relation_to_position_to_count_map[ent] = o_relation_to_pos_to_count_map
                o_relation_to_pos_to_count_map[relation_name] = {EntityDomainID.SUBJECT: count}
                if count > max_count_for_subj:
                    max_count_for_subj = count
                    max_subj_ent = ent

            print(f"{relation_name} counting Object")
            obj_ent_to_triple_count_map = self.count_n_triples_per_ent_in_rel_in_position(
                relation_df=relation_df,
                entity_domain=EntityDomainID.OBJECT,
                o_ent_to_triple_count_map=dict()
            )
            max_count_for_obj: int = 0
            max_obj_ent = None
            for ent, count in obj_ent_to_triple_count_map.items():
                o_relation_to_pos_to_count_map: Optional[
                    Dict[
                        RelationStr,
                        Dict[EntityDomainID, int]
                    ]
                ] = self.ent_to_relation_to_position_to_count_map.get(ent, None)
                if o_relation_to_pos_to_count_map is None:
                    o_relation_to_pos_to_count_map = dict()
                    self.ent_to_relation_to_position_to_count_map[ent] = o_relation_to_pos_to_count_map
                o_pos_to_count_map = o_relation_to_pos_to_count_map.get(relation_name)
                if o_pos_to_count_map is None:
                    o_pos_to_count_map = dict()
                    o_relation_to_pos_to_count_map[relation_name] = o_pos_to_count_map
                o_pos_to_count_map[EntityDomainID.OBJECT] = count
                if count > max_count_for_obj:
                    max_count_for_obj = count
                    max_obj_ent = ent

            position_to_max_count_map = {
                EntityDomainID.SUBJECT: (max_subj_ent, max_count_for_subj),
                EntityDomainID.OBJECT: (max_obj_ent, max_count_for_obj)
            }
            self.relation_to_position_to_max_count_map[relation_name] = position_to_max_count_map

    def get_all_relations_but(self, relations_to_exclude: Set[RelationStr]) -> Set[str]:
        return self.all_relations_set - relations_to_exclude

    @classmethod
    def count_n_triples_per_ent_in_rel_in_position(cls,
                                                   relation_df: pd.DataFrame,
                                                   entity_domain: EntityDomainID,
                                                   o_ent_to_triple_count_map: Optional[Dict[str, int]] = None
                                                   ) -> Dict[EntityStr, int]:

        if o_ent_to_triple_count_map is None:
            print("No counter given; initializing a new dict")
            o_ent_to_triple_count_map = dict()

        df_counts_per_pair = relation_df.groupby(by=[entity_domain.value], as_index=False).count()
        for row_i, row in df_counts_per_pair.iterrows():
            entity: EntityStr = row[entity_domain.value]
            n_triples: int = row[entity_domain.get_other().value]
            o_ent_to_triple_count_map[entity] = (
                    o_ent_to_triple_count_map.get(entity, 0) + n_triples
            )
        return o_ent_to_triple_count_map
