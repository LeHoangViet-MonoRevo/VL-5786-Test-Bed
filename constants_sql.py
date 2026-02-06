class ConstantSQL:
    DIMENSION_PRIORITY_COMMAND_OCR = """
        SELECT 
            d.physical_type_id,
            d.result
        FROM physical_obj.diagram_ocr d
        INNER JOIN (
            SELECT 
                physical_type_id,
                MAX(updated_at) AS max_updated_at
            FROM physical_obj.diagram_ocr
            WHERE organization_id = %s
                AND deleted_at IS NULL
                AND JSON_EXTRACT(result, '$.product_shape.dimension') IS NOT NULL
                AND physical_type_id IN %s
            GROUP BY physical_type_id
        ) t ON (
            d.physical_type_id = t.physical_type_id
            AND d.updated_at = t.max_updated_at
        );
    """

    SHAPE_PRIORITY_COMMAND_OCR = """
        SELECT 
            d.physical_type_id,
            d.result
        FROM physical_obj.diagram_ocr d
        INNER JOIN (
            SELECT 
                physical_type_id,
                MAX(updated_at) AS max_updated_at
            FROM physical_obj.diagram_ocr
            WHERE organization_id = %s
                AND deleted_at IS NULL
                AND JSON_EXTRACT(result, '$.product_shape.dimension') IS NOT NULL
                AND physical_type_id IN %s
            GROUP BY physical_type_id
        ) t ON (
            d.physical_type_id = t.physical_type_id
            AND d.updated_at = t.max_updated_at
        );
    """


constanst_sql = ConstantSQL()
