from osgeo import osr


def parse_wkt_info(wkt: str) -> dict:
    """
    解析 WKT 投影文本，返回关键信息字典：
      - projection: 投影名称
      - standard_parallel: 标准纬线
      - central_meridian: 中央经线
      - false_easting / northing: 假东，假北
      - linear_unit: 线性单位
    """
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return {
        'projection': srs.GetAttrValue('PROJECTION', 0),
        'standard_parallel': srs.GetNormProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0),
        'central_meridian': srs.GetNormProjParm(osr.SRS_PP_CENTRAL_MERIDIAN, 0.0),
        'false_easting': srs.GetNormProjParm(osr.SRS_PP_FALSE_EASTING, 0.0),
        'false_northing': srs.GetNormProjParm(osr.SRS_PP_FALSE_NORTHING, 0.0),
        'unit': srs.GetLinearUnitsName(),
    }
