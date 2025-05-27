from typing import Self
import polars as pl
from quantlab.arrays.main import ArrayBase
from quantlab.frames.graphs import FrameVisualizer
from quantlab.frames.main import FrameBase
from quantlab.frames.seasonality import SeasonalityExecutor
from quantlab.interfaces.types import Attributes



class Array(ArrayBase):
    """Container de type tableau pour opérations numériques et conversion en Frame.

    Inheritance:
        ArrayBase: fournit les opérations de base sur les tableaux.

    Attributes:
        __slots__ (tuple): définit les attributs de stockage interne depuis Attributes.DATA.
    """
    __slots__ = Attributes.DATA

    def to_df(self) -> "Frame":
        """Convertit cet Array en Frame pour traitement tabulaire.

        Effectue une collecte du LazyFrame sous-jacent et renvoie un nouvel objet Frame.

        Returns:
            Frame: instance contenant les mêmes données sous forme tabulaire.

        Example:
            >>> arr = Array(data=...)
            >>> frame = arr.to_df()
            >>> isinstance(frame, Frame)
            True
        """
        ...

    def to_lazyframe(self) -> pl.LazyFrame:
        """Produit un polars.LazyFrame pour calcul différé.

        - Oriente les données en lignes.
        - Remplit les NaN avec None.
        - Sélectionne les colonnes date et valeurs.

        Returns:
            pl.LazyFrame: représentation lazy du tableau.

        Example:
            >>> lazy = arr.to_lazyframe()
            >>> isinstance(lazy, pl.LazyFrame)
            True
        """
        ...


class Frame(FrameBase):
    """Structure tabulaire pour séries temporelles et analyses multidimensionnelles.

    Inheritance:
        FrameBase: fournit les opérations de base sur les frames.

    Attributes:
        __slots__ (tuple): définit les attributs de stockage interne depuis Attributes.DATA.
    """
    __slots__ = Attributes.DATA

    def clean_nans(self, total: bool = False) -> Self:
        """Supprime les lignes contenant des NaN.

        Args:
            total (bool): 
                - True : supprime toute ligne ayant au moins un NaN.  
                - False : supprime seulement les lignes dont toutes les valeurs sont NaN.  
                Default is False.

        Returns:
            Self: nouvelle instance de Frame sans les lignes NaN.

        Example:
            >>> cleaned = frame.clean_nans(total=True)
        """
        ...

    def to_array(self) -> Array:
        """Convertit cette Frame en Array pour calculs denses.

        Extrait la matrice de valeurs et l’index, puis les encapsule dans Array.

        Returns:
            Array: instance contenant les données de la Frame sous forme matricielle.

        Example:
            >>> arr = frame.to_array()
            >>> isinstance(arr, Array)
        """
        ...

    @property
    def plot(self) -> FrameVisualizer:
        """Accède au visualiseur pour générer des graphiques.

        Returns:
            FrameVisualizer: objet permettant de tracer la Frame.

        Example:
            >>> viz = frame.plot
            >>> viz.line()
        """
        ...

    def seasonal(self) -> SeasonalityExecutor[Self]:
        """Prépare une analyse de saisonnalité sur cette Frame.

        Retourne un exécuteur pour décomposer la série en tendance, saisonnalité, résidus, etc.

        Returns:
            SeasonalityExecutor[Self]: exécuteur chaînable pour analyse saisonnière.

        Example:
            >>> exe = frame.seasonal()
        """
        ...


def to_frame(df: pl.DataFrame, values_col: str, on: str) -> Frame:
    """Crée une Frame à partir d’un DataFrame Polars.

    Pivote le DataFrame selon la colonne `on` et les valeurs `values_col`.

    Args:
        df (pl.DataFrame): DataFrame contenant au moins les colonnes `on` et `values_col`.
        values_col (str): nom de la colonne à utiliser comme valeurs.
        on (str): nom de la colonne servant de pivot (génère plusieurs séries).

    Returns:
        Frame: instance représentant les données pivotées.

    Example:
        >>> frame = to_frame(df, values_col="value", on="asset")
    """
    ...


def to_array(df: pl.DataFrame, values_col: str, on: str) -> Array:
    """Crée un Array à partir d’un DataFrame Polars.

    Pivote, extrait la matrice de valeurs et l’index, puis enveloppe le tout dans un Array.

    Args:
        df (pl.DataFrame): DataFrame contenant les colonnes à pivoter.
        values_col (str): nom de la colonne des valeurs numériques.
        on (str): nom de la colonne de pivot.

    Returns:
        Array: Array contenant la matrice de valeurs et l’index.

    Example:
        >>> arr = to_array(df, values_col="value", on="asset")
    """
    ...