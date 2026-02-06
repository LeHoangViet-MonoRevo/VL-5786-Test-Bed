import json

import pandas as pd

from connection import mysql_physical_obj_connection
from constants_sql import constanst_sql


def get_diagram_ocr(
    organization_id, list_physical_id, mode: str = "dimension_priority"
):
    assert mode in ["dimension_priority", "shape_priority"]
    conn_physical = mysql_physical_obj_connection()

    ocr_command = (
        constanst_sql.DIMENSION_PRIORITY_COMMAND_OCR
        if mode == "dimension_priority"
        else constanst_sql.SHAPE_PRIORITY_COMMAND_OCR
    )
    ocr_df = pd.read_sql(
        ocr_command, conn_physical, params=(organization_id, list_physical_id)
    )
    # Autofill missing physical_type_id
    default_ocr = {
        "product_shape": {"shape": "", "dimension": "0x0x0"},
        "processing_content": {},
        "lathe_processing_content": {},
        "material_type": {},
        "required_precision": {},
        "surface_treatment": {},
        "heat_treatment": {},
        "basic_info": {},
    }
    found_ids = set(ocr_df["physical_type_id"].tolist()) if not ocr_df.empty else set()
    missing_ids = [pid for pid in list_physical_id if pid not in found_ids]
    if missing_ids:
        fill_df = pd.DataFrame(
            {
                "physical_type_id": missing_ids,
                "result": [json.dumps(default_ocr)] * len(missing_ids),
            }
        )
        ocr_df = pd.concat([ocr_df, fill_df], ignore_index=True)
    return ocr_df


def get_diagram_ocr_physical_types(organization_id):
    conn_physical = mysql_physical_obj_connection()

    query = """
        SELECT
            pt.id,
            pt.product_no,
            pt.product_name,
            pt.figure_number,
            ptf.location
        FROM physical_obj.physical_types pt
        JOIN (
            SELECT id, MAX(updated_at) AS max_updated_at
            FROM physical_obj.physical_types
            WHERE deleted_at IS NULL
            AND organization_id = %s
            AND (
                    product_no IS NOT NULL
                OR product_name IS NOT NULL
                OR figure_number IS NOT NULL
            )
            GROUP BY id
        ) latest
            ON pt.id = latest.id
            AND pt.updated_at = latest.max_updated_at
        JOIN physical_obj.physical_type_files ptf
            ON ptf.physical_type_id = pt.id
            AND ptf.deleted_at IS NULL
            AND ptf.is_raijin = 1
            AND ptf.location IS NOT NULL
    """

    ocr_df = pd.read_sql(query, conn_physical, params=[organization_id])
    if "customer_name" not in ocr_df.columns:
        ocr_df["customer_name"] = None
    return ocr_df
