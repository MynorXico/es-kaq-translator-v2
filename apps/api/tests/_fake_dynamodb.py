"""A minimal fake standing in for a `boto3` DynamoDB `Table` resource,
used across `tests/unit/test_jobs.py`, `tests/unit/test_worker_handler.py`,
`tests/unit/test_failure_handler.py`, and
`tests/integration/test_translate_jobs.py` -- mocked at the boundary
(`app.jobs`'s own `boto3.resource("dynamodb")` call), per
docs/testing.md's "mocked at the boundary" policy for apps/api, never
hitting real AWS (no `moto` dependency needed for this project's current
needs).

Deliberately not a full DynamoDB emulator: it only understands the exact
`UpdateExpression`/`ConditionExpression` shape `app.jobs._conditional_update`
actually sends (a `SET #a = :a, #b = :b, ...` update, optionally guarded by
an `attribute_not_exists(#x) OR #x = :y`-shaped condition), which is enough
to exercise the real conditional-write guard logic end-to-end rather than
having the test fake just assert whatever the implementation already does.
"""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError


class FakeDynamoDBTable:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def seed(self, item: dict[str, Any]) -> None:
        """Test setup helper: pre-populate an item, bypassing `put_item`."""
        self.items[item["job_id"]] = dict(item)

    def put_item(self, Item: dict[str, Any]) -> None:
        self.items[Item["job_id"]] = dict(Item)

    def get_item(self, Key: dict[str, Any]) -> dict[str, Any]:
        item = self.items.get(Key["job_id"])
        return {"Item": dict(item)} if item is not None else {}

    def update_item(
        self,
        *,
        Key: dict[str, Any],
        UpdateExpression: str,
        ExpressionAttributeNames: dict[str, str] | None = None,
        ExpressionAttributeValues: dict[str, Any] | None = None,
        ConditionExpression: str | None = None,
    ) -> None:
        job_id = Key["job_id"]
        item = self.items.setdefault(job_id, {"job_id": job_id})
        names = ExpressionAttributeNames or {}
        values = ExpressionAttributeValues or {}

        if ConditionExpression is not None and not self._evaluate_condition(
            ConditionExpression, item, names, values
        ):
            raise ClientError(
                {
                    "Error": {
                        "Code": "ConditionalCheckFailedException",
                        "Message": "The conditional request failed",
                    }
                },
                "UpdateItem",
            )

        assert UpdateExpression.startswith("SET "), (
            f"fake only supports a plain SET expression, got: {UpdateExpression!r}"
        )
        for assignment in UpdateExpression.removeprefix("SET ").split(", "):
            name_placeholder, value_placeholder = (part.strip() for part in assignment.split("="))
            attr_name = names[name_placeholder]
            item[attr_name] = values[value_placeholder]

    @staticmethod
    def _evaluate_condition(
        expr: str,
        item: dict[str, Any],
        names: dict[str, str],
        values: dict[str, Any],
    ) -> bool:
        # Only supports the "attribute_not_exists(#x) OR #x = :y" shape
        # app.jobs._conditional_update actually sends -- an OR of clauses,
        # true overall if any clause is true.
        for clause in (part.strip() for part in expr.split(" OR ")):
            if clause.startswith("attribute_not_exists("):
                name_placeholder = clause[len("attribute_not_exists(") : -1]
                attr_name = names[name_placeholder]
                if attr_name not in item:
                    return True
            elif "=" in clause:
                name_placeholder, value_placeholder = (part.strip() for part in clause.split("="))
                attr_name = names[name_placeholder]
                if item.get(attr_name) == values[value_placeholder]:
                    return True
        return False


class FakeDynamoDBResource:
    """Stands in for `boto3.resource("dynamodb")` -- `.Table(name)` always
    returns the same `FakeDynamoDBTable`, regardless of the name requested,
    since these tests only ever exercise one table at a time.
    """

    def __init__(self) -> None:
        self.table = FakeDynamoDBTable()

    def Table(self, _name: str) -> FakeDynamoDBTable:
        return self.table
