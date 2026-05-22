"""FireRed Agent: Multi-image preprocessing agent for FireRed-Image-Edit.

When users provide more than 3 images, this agent automatically:
1. Uses Gemini function calling to detect ROI regions in each image
2. Crops, resizes, and stitches images into 2-3 composite images
3. Rewrites (recaptions) user instructions to match the new image layout
"""

from agent.pipeline import AgentPipeline

__all__ = ["AgentPipeline"]
