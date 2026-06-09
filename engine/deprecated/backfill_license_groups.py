"""
Backfill BELONGS_TO per License senza LicenseGroup.
Eseguire una tantum dopo fix degli ID gruppo in mass_licences_other.py.
"""
from neo4j import GraphDatabase

from mass_licences_other import AUTH, URI, collega_licenza_al_gruppo, risolvi_gruppo_rischio


def esegui_backfill_gruppi():
    driver = GraphDatabase.driver(URI, auth=AUTH)

    query_licenze_orfane = """
    MATCH (l:License)
    WHERE NOT (l)-[:BELONGS_TO]->(:LicenseGroup)
    RETURN DISTINCT l.name AS license_name
    ORDER BY license_name
    """

    with driver.session() as session:
        license_names = [record["license_name"] for record in session.run(query_licenze_orfane)]

    print(f"Trovate {len(license_names)} licenze senza gruppo.\n")

    collegate = 0
    fallite = 0

    for license_name in license_names:
        group_id = risolvi_gruppo_rischio(license_name)

        try:
            with driver.session() as session:
                linked = collega_licenza_al_gruppo(session, license_name, group_id)

            if linked:
                collegate += 1
                print(f"[OK] {license_name} -> {group_id}")
            else:
                fallite += 1
                print(f"[WARN] {license_name} -> gruppo {group_id} non collegato (nodo gruppo assente?)")
        except ValueError as error:
            fallite += 1
            print(f"[ERRORE] {license_name}: {error}")

    driver.close()
    print(f"\n[COMPLETATO] Collegate: {collegate} | Fallite: {fallite}")


if __name__ == "__main__":
    esegui_backfill_gruppi()
