from __future__ import annotations

import json


import pandas as pd


from fedptha.data.tasks import TASKS
from fedptha.data.clients import parse_subset_clients


def _load_json_value(value, default):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(str(value))
    except Exception as exc:
        raise ValueError(f"Could not parse JSON value: {value}") from exc


def _json_dump(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def expand_oracle_hard_clusters(
    parent_df: pd.DataFrame,
) -> pd.DataFrame:
    """Split every 20-client oracle row into four 5-client rows.

    Each generated row contains exactly one true physical community:
    one client for every task and one shared base graph. APPLE-PostALA
    therefore runs as a fully independent five-client federation.
    """
    rows = []

    for parent_row_idx, parent_row in parent_df.iterrows():
        parent_clients = parse_subset_clients(parent_row["subset_clients"])

        if len(parent_clients) != 20:
            raise ValueError(
                f"Parent row {parent_row_idx}: expected 20 clients, "
                f"found {len(parent_clients)}: {parent_clients}"
            )

        membership = _load_json_value(
            parent_row.get("membership_json"),
            [],
        )

        if len(membership) != 20:
            raise ValueError(
                f"Parent row {parent_row_idx}: expected 20 "
                f"membership records, found {len(membership)}."
            )

        membership_by_community = {}

        for item in membership:
            if "planted_community_id" not in item:
                raise ValueError("membership_json has no planted_community_id.")

            community_id = int(item["planted_community_id"])

            membership_by_community.setdefault(
                community_id,
                [],
            ).append(dict(item))

        if len(membership_by_community) != 4:
            raise ValueError(
                f"Parent row {parent_row_idx}: expected four "
                f"communities, found "
                f"{sorted(membership_by_community)}."
            )

        parent_subset_id = str(
            parent_row.get(
                "subset_id",
                f"parent_{parent_row_idx}",
            )
        )
        parent_subset_clients = str(parent_row["subset_clients"])

        for community_id in sorted(membership_by_community):
            members = sorted(
                membership_by_community[community_id],
                key=lambda item: int(item["graph_id"]),
            )

            if len(members) != 5:
                raise ValueError(
                    f"Community {community_id}: expected five "
                    f"clients, found {len(members)}."
                )

            member_ids = [int(item["graph_id"]) for item in members]
            member_id_set = set(member_ids)

            if not member_id_set.issubset(set(parent_clients)):
                raise ValueError(
                    f"Community {community_id}: membership IDs "
                    f"{member_ids} are not contained in parent "
                    f"clients {parent_clients}."
                )

            assigned_tasks = [str(item["assigned_task"]) for item in members]

            if set(assigned_tasks) != set(TASKS):
                raise ValueError(
                    f"Community {community_id}: expected one "
                    f"client for every task {TASKS}, found "
                    f"{assigned_tasks}."
                )

            base_graph_ids = {int(item["base_graph_id"]) for item in members}

            if len(base_graph_ids) != 1:
                raise ValueError(
                    f"Community {community_id}: expected one "
                    f"base graph, found {base_graph_ids}."
                )

            base_dataset_ids = sorted(
                {
                    str(item["base_dataset_id"])
                    for item in members
                    if item.get("base_dataset_id") is not None
                }
            )

            graph_to_task = {
                str(item["graph_id"]): str(item["assigned_task"]) for item in members
            }
            graph_to_community = {
                str(item["graph_id"]): int(item["planted_community_id"])
                for item in members
            }
            graph_to_family = {
                str(item["graph_id"]): (f"q_task_{item['assigned_task']}")
                for item in members
            }

            family_to_graph_ids = {
                f"q_task_{item['assigned_task']}": [int(item["graph_id"])]
                for item in members
            }

            family_counts = {
                family: len(graph_ids)
                for family, graph_ids in family_to_graph_ids.items()
            }

            hard_row = parent_row.to_dict()

            hard_row["oracle_parent_subset_id"] = parent_subset_id
            hard_row["oracle_parent_subset_clients"] = parent_subset_clients
            hard_row["oracle_hard_cluster_id"] = int(community_id)
            hard_row["oracle_hard_base_graph_id"] = int(next(iter(base_graph_ids)))
            hard_row["oracle_hard_community_source"] = (
                "membership_json.planted_community_id"
            )

            hard_row["family"] = "oracle_hard_cluster_q_task_label_heterogeneity"
            hard_row["subset_id"] = (
                f"{parent_subset_id}" f"__oracle_hard_c{community_id}"
            )
            hard_row["subset_clients"] = "|".join(
                str(graph_id) for graph_id in member_ids
            )
            hard_row["subset_size"] = 5

            hard_row["num_structural_clusters"] = 1
            hard_row["clients_per_structural_cluster"] = 5

            hard_row["graph_ids_json"] = _json_dump(member_ids)
            hard_row["dataset_ids_json"] = _json_dump(
                [str(item["dataset_id"]) for item in members]
            )
            hard_row["base_graph_ids_json"] = _json_dump(sorted(base_graph_ids))
            hard_row["base_dataset_ids_json"] = _json_dump(base_dataset_ids)
            hard_row["membership_json"] = _json_dump(members)
            hard_row["graph_to_task_json"] = _json_dump(graph_to_task)
            hard_row["graph_to_community_json"] = _json_dump(graph_to_community)
            hard_row["graph_to_family_json"] = _json_dump(graph_to_family)
            hard_row["community_to_graph_ids_json"] = _json_dump(
                {
                    str(community_id): member_ids,
                }
            )
            hard_row["family_to_graph_ids_json"] = _json_dump(family_to_graph_ids)
            hard_row["family_counts_json"] = _json_dump(family_counts)

            rows.append(hard_row)

    expanded = pd.DataFrame(rows)

    if expanded.empty:
        raise ValueError("Oracle hard-cluster expansion produced no rows.")

    return expanded.reset_index(drop=True)
