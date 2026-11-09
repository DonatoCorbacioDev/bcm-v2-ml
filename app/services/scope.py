# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

"""The one rule every data query in this service follows: it is scoped to an organization."""


def require_org(org_id: int | None) -> int:
    """Returns org_id, or refuses when there is none.

    A missing organization never means "every organization": the service fails
    closed, exactly like the backend's TenantContext.require(). A manager_id,
    when present, narrows the organization further; it never replaces it.
    """
    if org_id is None:
        raise ValueError("org_id is required: every query is scoped to one organization")
    return org_id
