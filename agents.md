# NarratoAI - AI Video Narration and Editing Tool

## Project Overview

NarratoAI is an automated video narration tool that provides an all-in-one solution for script writing, automated video editing, voice-over (TTS), and subtitle generation. It uses Large Language Models (LLMs) to understand video content and generate professional narration scripts.

### Core Features

1. **Video Frame Analysis**: Extracts keyframes from videos and analyzes them using vision models
2. **Script Generation**: Generates narration scripts based on video content analysis
3. **Video Editing**: Automatically clips and merges video segments based on scripts
4. **Text-to-Speech (TTS)**: Converts narration scripts to audio using various TTS engines
5. **Subtitle Generation**: Automatically generates subtitles synchronized with narration
6. **Background Music**: Supports adding background music to videos

## Project Architecture

### Directory Structure

```
NarratoAI/
├── app/                    # Core application code
│   ├── config/            # Configuration management
│   ├── models/            # Data models and schemas
│   ├── services/          # Business logic services
│   │   ├── llm/          # LLM service providers (LiteLLM unified interface)
│   │   ├── prompts/     # Prompt templates management
│   │   ├── task.py      # Main video generation workflow
│   │   └── ...
│   └── utils/            # Utility functions
├── webui/                 # Streamlit web interface
│   ├── components/       # UI components
│   ├── tools/           # Script generation tools
│   └── utils/           # UI utilities
├── resource/            # Resource files (videos, songs, scripts, fonts)
├── storage/             # Temporary files and cache
├── config.toml         # Main configuration file
└── webui.py            # Main entry point
```

### Key Components

#### 1. LLM Service Layer (`app/services/llm/`)

- **Unified Interface**: Uses LiteLLM to support 100+ LLM providers
- **Vision Models**: For analyzing video frames (e.g., Gemini, OpenAI, Qwen, OpenRouter)
- **Text Models**: For generating narration scripts (e.g., DeepSeek, OpenAI, Gemini, OpenRouter)
- **Migration Adapter**: Provides backward compatibility with legacy code

#### 2. Prompt Management (`app/services/prompts/`)

- **Template System**: Manages prompt templates for different use cases
- **Categories**: 
  - `documentary/`: For documentary-style narration
  - `short_drama_narration/`: For short drama narration
  - `short_drama_editing/`: For short drama editing

#### 3. Video Processing (`app/services/`)

- **Script Generation**: `script_service.py` - Generates video scripts from keyframes
- **Video Clipping**: `clip_video.py` - Clips video segments based on timestamps
- **Video Merging**: `merger_video.py` - Merges video segments
- **TTS Generation**: `voice.py` - Text-to-speech conversion
- **Subtitle Generation**: `subtitle.py` - Subtitle creation

#### 4. Web UI (`webui/`)

- **Streamlit Interface**: User-friendly web interface
- **Components**: Modular UI components for different settings
- **Tools**: Script generation tools for different video types

## Configuration

### Main Configuration File: `config.toml`

The project uses TOML format for configuration. Key sections:

#### LLM Configuration

```toml
[app]
# Vision Model (for analyzing video frames)
vision_llm_provider = "litellm"
vision_litellm_model_name = "openrouter/allenai/molmo-2-8b:free"
vision_litellm_api_key = "your-api-key"
vision_litellm_base_url = "https://openrouter.ai/api/v1"

# Text Model (for generating narration scripts)
text_llm_provider = "litellm"
text_litellm_model_name = "openrouter/deepseek/deepseek-chat"
text_litellm_api_key = "your-api-key"
text_litellm_base_url = "https://openrouter.ai/api/v1"
```

#### TTS Configuration

```toml
[ui]
tts_engine = "edge_tts"  # Options: edge_tts, azure_speech, tencent_tts, soulvoice, tts_qwen, indextts2
edge_voice_name = "zh-CN-XiaoyiNeural-Female"
```

#### Video Processing Configuration

```toml
[frames]
frame_interval_input = 3      # Keyframe extraction interval (seconds)
vision_batch_size = 10        # Batch size for vision model processing
```

## Security Notice

- Do not commit or store any sensitive information in the repository (e.g., API keys, tokens, passwords, or private URLs).

## Workflow

### 1. Documentary Video Script Generation

**Path**: `webui/tools/generate_script_docu.py`

**Process**:
1. **Keyframe Extraction**: Extracts keyframes from video at specified intervals
2. **Vision Analysis**: Uses vision model to analyze each keyframe batch
3. **Frame Analysis Processing**: Converts analysis results to Markdown format
4. **Narration Generation**: Uses text model to generate narration script from frame analysis
5. **Script Formatting**: Outputs JSON format with timestamps, picture descriptions, and narration

**Key Functions**:
- `generate_script_docu(params)`: Main entry point
- Uses `app/services/generate_narration_script.py` for narration generation
- Supports custom prompts via `custom_prompt` parameter

### 2. Video Generation Pipeline

**Path**: `app/services/task.py`

**Process**:
1. **Load Script**: Loads the narration script JSON file
2. **TTS Generation**: Generates audio files for narration segments (OST=0 or OST=2)
3. **Video Clipping**: Clips video segments based on timestamps in script
4. **Audio Merging**: Merges narration audio, original audio (if OST=2), and background music
5. **Subtitle Generation**: Creates subtitles synchronized with narration
6. **Final Video Assembly**: Combines all elements into final video

**Key Functions**:
- `start_subclip_unified(task_id, params)`: Main video generation function
- Handles OST types:
  - `OST=0`: Narration only (no original audio)
  - `OST=1`: Original audio only (no narration)
  - `OST=2`: Both narration and original audio

### 3. Script Format

The narration script is a JSON array with the following structure:

```json
[
  {
    "_id": 1,
    "timestamp": "00:00:00,000-00:00:10,000",
    "picture": "Description of the video frame",
    "narration": "Narration text for this segment",
    "OST": 2
  }
]
```

**Fields**:
- `_id`: Segment ID
- `timestamp`: Time range in format `HH:MM:SS,mmm-HH:MM:SS,mmm`
- `picture`: Description of the video content
- `narration`: Narration text (or "播放原片+N" for original audio segments)
- `OST`: Audio type (0=narration only, 1=original only, 2=both)

## Usage Guide

### Initial Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   - Copy `config.example.toml` to `config.toml`
   - Fill in your LLM API keys (OpenRouter, OpenAI, Gemini, etc.)
   - Configure TTS settings if needed

3. **Start the Application**:
   ```bash
   streamlit run webui.py --server.maxUploadSize=2048
   ```

### Basic Workflow

1. **Upload Video**: 
   - Go to "Script Settings" section
   - Upload your video file (supports mp4, mov, avi, flv, mkv)

2. **Generate Script**:
   - Select script type: "Auto Generate" (for documentaries)
   - Optionally provide:
     - Video Theme: Brief description of video content
     - Generation Prompt: Custom requirements (e.g., "Focus on goal scenes, end with sweaty finish")
   - Click "Generate Video Script"
   - Wait for keyframe extraction and analysis
   - Review generated script in the text area

3. **Edit Script** (Optional):
   - Modify the JSON script directly in the text area
   - Adjust timestamps, narration text, or OST values
   - Click "Save Script" to save changes

4. **Configure Settings**:
   - **Audio Settings**: Choose TTS engine, voice, volume, rate, pitch
   - **Video Settings**: Output resolution, frame rate, etc.
   - **Subtitle Settings**: Font, size, color, position

5. **Generate Video**:
   - Click "Generate Video" button
   - Monitor progress in the progress bar
   - Download the final video when complete

### Advanced Features

#### Custom Prompts

You can provide custom prompts in the "Generation Prompt" field to guide script generation:

```
视频内容：我每周末六点早起打篮球，从热身到比赛，最后大汗淋漓。

重点要求：
1. 重点提取和突出进球的精彩镜头
2. 每个进球都要有精彩的解说
3. 结尾要展现大汗淋漓的状态，并设计一个激励人心的结尾
```

The system will integrate your custom requirements into the prompt template.

#### Background Music

- Upload multiple background music files in "Audio Settings"
- Files are saved to `resource/songs/`
- The system can randomly select or use specific BGM files

#### Multiple Video Types

- **Documentary**: Auto-generate script from video frames (no subtitles/audio in original)
- **Short Drama**: Mix and edit short drama videos
- **Short Drama Narration**: Generate narration for short dramas with subtitles

## LLM Provider Configuration

### Supported Providers (via LiteLLM)

- **OpenRouter**: `openrouter/model-id` (e.g., `openrouter/deepseek/deepseek-chat`)
- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`
- **Gemini**: `gemini/gemini-2.0-flash-lite`
- **DeepSeek**: `deepseek/deepseek-chat`
- **Qwen**: `qwen/qwen-plus`
- **And 100+ more providers**

### Model Name Format

For LiteLLM, use format: `provider/model-id`

Examples:
- `openrouter/deepseek/deepseek-chat`
- `gemini/gemini-2.0-flash-lite`
- `openai/gpt-4o-mini`

### OpenRouter Special Handling

- OpenRouter uses OpenAI-compatible API
- Model names should be `openrouter/model-id` in config (e.g., `openrouter/deepseek/deepseek-chat`)
- API automatically removes `openrouter/` prefix when calling (uses `deepseek/deepseek-chat` internally)
- Supports both free and paid models
- **Important**: When using OpenAI SDK directly, model ID should NOT include `openrouter/` prefix
- The system automatically handles model name conversion for OpenRouter

### Recent Fixes (2025-01-17)

1. **OpenRouter API Key Configuration**: Fixed API key passing for OpenRouter models
2. **Model Name Format**: Fixed model ID format when calling OpenRouter API (removes `openrouter/` prefix)
3. **Custom Prompt Integration**: Custom prompts from "Generation Prompt" field are now properly integrated
4. **Error Handling**: Improved error handling with automatic fallback for unsupported parameters
5. **Direct OpenAI SDK Support**: Added direct OpenAI SDK calls for OpenRouter (more reliable than LiteLLM)

## File Paths

### Resource Directories

- **Videos**: `resource/videos/` - Upload video files here
- **Background Music**: `resource/songs/` - Upload BGM files here
- **Scripts**: `resource/scripts/` - Saved narration scripts (JSON format)
- **Fonts**: `resource/fonts/` - Font files for subtitles

### Temporary Files

- **Keyframes**: `storage/temp/keyframes/` - Extracted video keyframes
- **Analysis**: `storage/temp/analysis/` - Frame analysis results
- **Tasks**: `storage/tasks/` - Task state and progress

## Error Handling

### Common Issues

1. **FFmpeg Not Found**:
   - Install FFmpeg: `brew install ffmpeg` (macOS) or download for Windows
   - Set `ffmpeg_path` in `config.toml`

2. **LLM API Errors**:
   - Check API key validity
   - Verify model name format
   - Check account balance (for paid models)
   - Review error logs for detailed messages

3. **Script Generation Failures**:
   - Ensure video file is valid
   - Check if vision model supports video/image analysis
   - Verify text model supports JSON output
   - Try different models if current one fails

4. **JSON Parsing Errors**:
   - System automatically attempts to fix common JSON issues
   - Check if model output is valid JSON
   - Review system prompt for JSON format requirements

## API Integration

### Adding New LLM Providers

1. **Via LiteLLM**: Add provider name and API key to config
2. **Model Format**: Use `provider/model-id` format
3. **API Base URL**: Set if provider uses custom endpoint

### Custom Prompt Templates

Edit prompt templates in `app/services/prompts/`:
- `documentary/narration_generation.py`: Documentary narration prompts
- `short_drama_narration/script_generation.py`: Short drama narration prompts

## Development Notes

### Key Design Patterns

1. **Unified LLM Interface**: All LLM calls go through LiteLLM for consistency
2. **Migration Adapter**: Provides backward compatibility layer
3. **Prompt Management**: Centralized prompt template system
4. **Async Processing**: Uses asyncio for concurrent operations

### Testing

- Test model connections via "Test Connection" buttons in UI
- Check logs in console for detailed error information
- Verify script format before video generation

## Best Practices

1. **Model Selection**:
   - Use paid models for production (more stable, better JSON output)
   - Free models may have rate limits and quality issues
   - Test model compatibility before large-scale use

2. **Custom Prompts**:
   - Be specific about requirements
   - Mention key scenes or moments to highlight
   - Specify desired ending style

3. **Script Editing**:
   - Review generated scripts before video generation
   - Ensure timestamps don't overlap
   - Verify OST values are correct

4. **Performance**:
   - Adjust `frame_interval_input` to balance quality and cost
   - Use appropriate `vision_batch_size` for your model
   - Consider video length when setting timeouts

## Troubleshooting

### Debug Mode

Enable detailed logging by setting log level in `config.toml`:
```toml
log_level = "DEBUG"
```

### Common Solutions

1. **Model Connection Issues**:
   - Verify API key is correct
   - Check model name format
   - Ensure base_url is correct for provider

2. **JSON Output Issues**:
   - System automatically retries without `response_format` if needed
   - Check if model supports JSON mode
   - Review error logs for specific issues

3. **Video Generation Failures**:
   - Ensure script file exists and is valid JSON
   - Check video file path is correct
   - Verify FFmpeg is installed and accessible

## Future Enhancements

- Export to video editing software drafts
- Character face matching
- Automatic material matching based on narration
- Support for more TTS engines
- Multi-language support improvements
