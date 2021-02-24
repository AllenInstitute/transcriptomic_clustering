from argschema.schemas import ArgSchema, DefaultSchema
from argschema.fields import (
    InputFile, List, Float, Int, Nested
)
import anndata as ad

class AnnDataSchema(DefaultSchema):
    """
    AnnData: annotated data
    """
    adata = ad.AnnData()


class InputParameters(ArgSchema):
    cell_expression_path = InputFile(
        description=(
            "File at this location is a cell expressions (counts)."
        ),
        required=True
    )

class OutputParameters(DefaultSchema):
    normalized_cell_expressions = Nested(AnnDataSchema,
                                    required=False,
                                    description='AnnData holds the normalized cell expressions')
    