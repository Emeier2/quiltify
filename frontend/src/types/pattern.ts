export interface FabricSchema {
  id: string
  color_hex: string
  name: string
  total_sqin: number
  fat_quarters?: number
}

export interface BlockSchema {
  x: number
  y: number
  width: number
  height: number
  fabric_id: string
}

export interface QuiltPatternSchema {
  grid_width: number
  grid_height: number
  block_size_in: number
  seam_allowance: number
  fabrics: FabricSchema[]
  blocks: BlockSchema[]
  finished_width_in?: number
  finished_height_in?: number
}

export interface CutPiece {
  fabric_id: string
  fabric_name: string
  color_hex: string
  cut_width_in: number
  cut_height_in: number
  quantity: number
}

export interface GenerateResponse {
  pattern_json: QuiltPatternSchema
  svg: string
  cutting_svg: string
  cutting_chart: CutPiece[]
  guide: string
  confidence_score: number
  validation_errors: string[]
  pipeline_status: { loaded: boolean; type: string }
  image_b64: string | null
}

export interface QuiltifyResponse {
  pattern_json: QuiltPatternSchema
  svg: string
  cutting_svg: string
  cutting_chart: CutPiece[]
  guide: string
  confidence_score: number
  validation_errors: string[]
  original_image_b64: string
  quilt_image_b64: string | null
}

export interface GuideResponse {
  guide: string
  cutting_chart: CutPiece[]
  svg: string
  cutting_svg: string
  pattern_json: QuiltPatternSchema
  validation_errors: string[]
}
