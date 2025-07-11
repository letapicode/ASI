"""Expose MemoryServer via a minimal GraphQL API."""

from __future__ import annotations

from typing import List

import torch

from graphql import (
    GraphQLSchema,
    GraphQLObjectType,
    GraphQLField,
    GraphQLList,
    GraphQLFloat,
    GraphQLString,
    graphql_sync,
)

from .remote_memory import query_remote, push_remote


class GraphQLMemoryGateway:
    def __init__(self, address: str) -> None:
        self.address = address
        self.schema = GraphQLSchema(
            query=GraphQLObjectType(
                name="Query",
                fields={
                    "query": GraphQLField(
                        GraphQLList(GraphQLFloat),
                        args={"vector": GraphQLList(GraphQLFloat), "k": GraphQLFloat},
                        resolve=lambda obj, info, vector, k=5: query_remote(
                            address, torch.tensor(vector), int(k)
                        )[0]
                        .view(-1)
                        .tolist(),
                    )
                },
            ),
            mutation=GraphQLObjectType(
                name="Mutation",
                fields={
                    "push": GraphQLField(
                        GraphQLString,
                        args={"vector": GraphQLList(GraphQLFloat)},
                        resolve=lambda obj, info, vector: str(
                            push_remote(address, torch.tensor(vector))
                        ),
                    )
                },
            ),
        )

    def execute(self, query: str) -> dict:
        return graphql_sync(self.schema, query).data


__all__ = ["GraphQLMemoryGateway"]
