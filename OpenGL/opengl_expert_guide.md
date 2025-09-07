        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
        
        // Depth attachment
        glGenTextures(1, &depthTexture);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);
        
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "Framebuffer not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    void bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        glViewport(0, 0, width, height);
    }
    
    void unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    unsigned int getColorTexture() { return colorTexture; }
    unsigned int getDepthTexture() { return depthTexture; }
    
    ~Framebuffer() {
        glDeleteFramebuffers(1, &FBO);
        glDeleteTextures(1, &colorTexture);
        glDeleteTextures(1, &depthTexture);
    }
};

// Multi-target framebuffer for deferred rendering
class GBuffer {
private:
    unsigned int gBuffer;
    unsigned int gPosition, gNormal, gAlbedoSpec;
    unsigned int rboDepth;
    int width, height;
    
public:
    GBuffer(int w, int h) : width(w), height(h) {
        glGenFramebuffers(1, &gBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
        
        // Position buffer
        glGenTextures(1, &gPosition);
        glBindTexture(GL_TEXTURE_2D, gPosition);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
        
        // Normal buffer
        glGenTextures(1, &gNormal);
        glBindTexture(GL_TEXTURE_2D, gNormal);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
        
        // Albedo + Specular buffer
        glGenTextures(1, &gAlbedoSpec);
        glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedoSpec, 0);
        
        // Tell OpenGL which color attachments we'll use for rendering
        unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
        glDrawBuffers(3, attachments);
        
        // Depth buffer
        glGenRenderbuffers(1, &rboDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
        
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "Framebuffer not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    void bindForWriting() {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gBuffer);
    }
    
    void bindForReading() {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
    }
};
```

### Post-Processing Effects:
```cpp
// HDR and Tone Mapping
const char* hdrFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D hdrBuffer;
uniform bool hdr;
uniform float exposure;

void main() {             
    vec3 hdrColor = texture(hdrBuffer, TexCoords).rgb;
    if(hdr) {
        // Reinhard tone mapping
        vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
        // Exposure tone mapping
        // vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);
        
        // Gamma correction 
        mapped = pow(mapped, vec3(1.0 / 2.2));
  
        FragColor = vec4(mapped, 1.0);
    } else {
        vec3 result = pow(hdrColor, vec3(1.0 / 2.2));
        FragColor = vec4(result, 1.0);
    }
}
)";

// Bloom Effect
const char* bloomFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D scene;
uniform sampler2D bloomBlur;
uniform bool bloom;
uniform float exposure;

void main() {             
    vec3 hdrColor = texture(scene, TexCoords).rgb;      
    vec3 bloomColor = texture(bloomBlur, TexCoords).rgb;
    if(bloom)
        hdrColor += bloomColor; // additive blending
    
    // Tone mapping
    vec3 result = vec3(1.0) - exp(-hdrColor * exposure);
    // Gamma correction
    result = pow(result, vec3(1.0 / 2.2));
    FragColor = vec4(result, 1.0);
}
)";

// Gaussian Blur for Bloom
const char* blurFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D image;
uniform bool horizontal;
uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {             
    vec2 tex_offset = 1.0 / textureSize(image, 0);
    vec3 result = texture(image, TexCoords).rgb * weight[0];
    if(horizontal) {
        for(int i = 1; i < 5; ++i) {
            result += texture(image, TexCoords + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
            result += texture(image, TexCoords - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
        }
    } else {
        for(int i = 1; i < 5; ++i) {
            result += texture(image, TexCoords + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
            result += texture(image, TexCoords - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
        }
    }
    FragColor = vec4(result, 1.0);
}
)";

// Screen-space quad for post-processing
class ScreenQuad {
private:
    unsigned int quadVAO = 0;
    unsigned int quadVBO;
    
public:
    void renderQuad() {
        if (quadVAO == 0) {
            float quadVertices[] = {
                // positions        // texture Coords
                -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
                -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
                 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
                 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
            };
            
            glGenVertexArrays(1, &quadVAO);
            glGenBuffers(1, &quadVBO);
            glBindVertexArray(quadVAO);
            glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        }
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }
};
```

---

## 12. Geometry and Tessellation Shaders {#geometry-tessellation}

### Geometry Shader for Point Sprites:
```glsl
// Geometry shader for converting points to quads
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in VS_OUT {
    vec3 color;
} gs_in[];

out vec3 fColor;

uniform mat4 projection;

void build_house(vec4 position) {    
    fColor = gs_in[0].color; // gs_in[0] since there's only one input vertex
    gl_Position = position + vec4(-0.2, -0.2, 0.0, 0.0); // 1:bottom-left   
    EmitVertex();   
    gl_Position = position + vec4( 0.2, -0.2, 0.0, 0.0); // 2:bottom-right
    EmitVertex();
    gl_Position = position + vec4(-0.2,  0.2, 0.0, 0.0); // 3:top-left
    EmitVertex();
    gl_Position = position + vec4( 0.2,  0.2, 0.0, 0.0); // 4:top-right
    EmitVertex();
    EndPrimitive();
}

void main() {    
    build_house(gl_in[0].gl_Position);
}
```

### Tessellation Shaders for Terrain:
```glsl
// Tessellation Control Shader
#version 400 core

layout (vertices = 3) out;

in vec2 TextureCoord[];
out vec2 TextureCoord_TC_out[];

uniform float TessLevelInner;
uniform float TessLevelOuter;

void main() {
    TextureCoord_TC_out[gl_InvocationID] = TextureCoord[gl_InvocationID];
    
    if (gl_InvocationID == 0) {
        gl_TessLevelInner[0] = TessLevelInner;
        gl_TessLevelOuter[0] = TessLevelOuter;
        gl_TessLevelOuter[1] = TessLevelOuter;
        gl_TessLevelOuter[2] = TessLevelOuter;
    }
    
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}

// Tessellation Evaluation Shader
#version 400 core

layout(triangles, equal_spacing, cw) in;

uniform mat4 mvp;
uniform sampler2D heightMap;

in vec2 TextureCoord_TC_out[];
out float Height;

void main() {
    vec2 tc1 = gl_TessCoord.x * TextureCoord_TC_out[0];
    vec2 tc2 = gl_TessCoord.y * TextureCoord_TC_out[1];
    vec2 tc3 = gl_TessCoord.z * TextureCoord_TC_out[2];
    vec2 tc = normalize(tc1 + tc2 + tc3);
    
    vec4 p1 = gl_TessCoord.x * gl_in[0].gl_Position;
    vec4 p2 = gl_TessCoord.y * gl_in[1].gl_Position;
    vec4 p3 = gl_TessCoord.z * gl_in[2].gl_Position;
    vec4 pos = normalize(p1 + p2 + p3);
    
    Height = texture(heightMap, tc).r * 64.0 - 16.0;
    pos.y += Height;
    
    gl_Position = mvp * pos;
}
```

---

## 13. Compute Shaders {#compute-shaders}

### Basic Compute Shader Setup:
```cpp
class ComputeShader {
private:
    unsigned int ID;
    
public:
    ComputeShader(const char* computePath) {
        std::string computeCode;
        std::ifstream cShaderFile;
        cShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        
        try {
            cShaderFile.open(computePath);
            std::stringstream cShaderStream;
            cShaderStream << cShaderFile.rdbuf();
            cShaderFile.close();
            computeCode = cShaderStream.str();
        } catch(std::ifstream::failure& e) {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ" << std::endl;
        }
        
        const char* cShaderCode = computeCode.c_str();
        
        unsigned int compute = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(compute, 1, &cShaderCode, NULL);
        glCompileShader(compute);
        checkCompileErrors(compute, "COMPUTE");
        
        ID = glCreateProgram();
        glAttachShader(ID, compute);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");
        
        glDeleteShader(compute);
    }
    
    void use() {
        glUseProgram(ID);
    }
    
    void dispatch(unsigned int numGroupsX, unsigned int numGroupsY, unsigned int numGroupsZ) {
        glDispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
    }
    
    void wait() {
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
};

// Particle system with compute shader
const char* particleComputeShader = R"(
#version 430 core
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) restrict buffer PositionBuffer {
    vec4 positions[];
};

layout(std430, binding = 1) restrict buffer VelocityBuffer {
    vec4 velocities[];
};

uniform float deltaTime;
uniform float time;

void main() {
    uint index = gl_GlobalInvocationID.x;
    
    if (index >= positions.length()) {
        return;
    }
    
    vec3 position = positions[index].xyz;
    vec3 velocity = velocities[index].xyz;
    
    // Apply gravity
    velocity += vec3(0.0, -9.81, 0.0) * deltaTime;
    
    // Update position
    position += velocity * deltaTime;
    
    // Simple collision with ground plane
    if (position.y < 0.0) {
        position.y = 0.0;
        velocity.y = -velocity.y * 0.8; // damping
    }
    
    positions[index] = vec4(position, 1.0);
    velocities[index] = vec4(velocity, 0.0);
}
)";

// Usage example
void setupParticleSystem() {
    const int numParticles = 10000;
    
    // Generate initial particle data
    std::vector<glm::vec4> positions(numParticles);
    std::vector<glm::vec4> velocities(numParticles);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < numParticles; ++i) {
        positions[i] = glm::vec4(dis(gen) * 10.0f, dis(gen) * 10.0f + 10.0f, dis(gen) * 10.0f, 1.0f);
        velocities[i] = glm::vec4(dis(gen), dis(gen), dis(gen), 0.0f);
    }
    
    // Create SSBOs
    unsigned int positionSSBO, velocitySSBO;
    glGenBuffers(1, &positionSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * sizeof(glm::vec4), 
                 positions.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionSSBO);
    
    glGenBuffers(1, &velocitySSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocitySSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * sizeof(glm::vec4), 
                 velocities.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velocitySSBO);
}

// In render loop
void updateParticles(ComputeShader& computeShader, float deltaTime) {
    computeShader.use();
    computeShader.setFloat("deltaTime", deltaTime);
    computeShader.setFloat("time", glfwGetTime());
    
    // Dispatch compute shader (256 threads per group)
    computeShader.dispatch((numParticles + 255) / 256, 1, 1);
    computeShader.wait();
}
```

### Compute Shader for Image Processing:
```glsl
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D imgInput;
layout(rgba32f, binding = 1) uniform image2D imgOutput;

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    
    // Sobel edge detection
    vec4 tl = imageLoad(imgInput, pixel_coords + ivec2(-1, -1));   // top left
    vec4 tm = imageLoad(imgInput, pixel_coords + ivec2( 0, -1));   // top middle
    vec4 tr = imageLoad(imgInput, pixel_coords + ivec2( 1, -1));   // top right
    vec4 ml = imageLoad(imgInput, pixel_coords + ivec2(-1,  0));   // middle left
    vec4 mm = imageLoad(imgInput, pixel_coords + ivec2( 0,  0));   // middle middle
    vec4 mr = imageLoad(imgInput, pixel_coords + ivec2( 1,  0));   // middle right
    vec4 bl = imageLoad(imgInput, pixel_coords + ivec2(-1,  1));   // bottom left
    vec4 bm = imageLoad(imgInput, pixel_coords + ivec2( 0,  1));   // bottom middle
    vec4 br = imageLoad(imgInput, pixel_coords + ivec2( 1,  1));   // bottom right
    
    vec4 x = tl + 2.0*ml + bl - tr - 2.0*mr - br;
    vec4 y = tl + 2.0*tm + tr - bl - 2.0*bm - br;
    
    vec4 color = sqrt(x*x + y*y);
    
    imageStore(imgOutput, pixel_coords, color);
}
```

---

## 14. Performance Optimization {#optimization}

### GPU Profiling and Debugging:
```cpp
class GPUTimer {
private:
    unsigned int queryID[2];
    unsigned int queryBackBuffer = 0;
    unsigned int queryFrontBuffer = 1;
    
public:
    GPUTimer() {
        glGenQueries(2, queryID);
    }
    
    void start() {
        glBeginQuery(GL_TIME_ELAPSED, queryID[queryBackBuffer]);
    }
    
    void stop() {
        glEndQuery(GL_TIME_ELAPSED);
        std::swap(queryFrontBuffer, queryBackBuffer);
    }
    
    bool available() {
        int done = 0;
        glGetQueryObjectiv(queryID[queryFrontBuffer], GL_QUERY_RESULT_AVAILABLE, &done);
        return done == GL_TRUE;
    }
    
    float getTimeInMS() {
        GLuint64 timeElapsed = 0;
        glGetQueryObjectui64v(queryID[queryFrontBuffer], GL_QUERY_RESULT, &timeElapsed);
        return timeElapsed * 1e-6f; // Convert to milliseconds
    }
    
    ~GPUTimer() {
        glDeleteQueries(2, queryID);
    }
};

// Usage
GPUTimer timer;
timer.start();
// ... render operations ...
timer.stop();

// Later, check if results are available
if (timer.available()) {
    float renderTime = timer.getTimeInMS();
    std::cout << "Render time: " << renderTime << " ms" << std::endl;
}
```

### Memory Management and Buffer Optimization:
```cpp
class BufferManager {
private:
    struct BufferAllocation {
        unsigned int buffer;
        size_t offset;
        size_t size;
        bool inUse;
    };
    
    std::vector<BufferAllocation> allocations;
    unsigned int masterBuffer;
    size_t bufferSize;
    size_t currentOffset = 0;
    
public:
    BufferManager(size_t size) : bufferSize(size) {
        glGenBuffers(1, &masterBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, masterBuffer);
        glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    }
    
    BufferAllocation* allocate(size_t size) {
        // Find free space or reuse
        for (auto& alloc : allocations) {
            if (!alloc.inUse && alloc.size >= size) {
                alloc.inUse = true;
                return &alloc;
            }
        }
        
        // Create new allocation
        if (currentOffset + size <= bufferSize) {
            BufferAllocation newAlloc;
            newAlloc.buffer = masterBuffer;
            newAlloc.offset = currentOffset;
            newAlloc.size = size;
            newAlloc.inUse = true;
            
            allocations.push_back(newAlloc);
            currentOffset += size;
            
            return &allocations.back();
        }
        
        return nullptr; // Out of memory
    }
    
    void deallocate(BufferAllocation* alloc) {
        if (alloc) {
            alloc->inUse = false;
        }
    }
};

// Vertex Array Object caching
class VAOCache {
private:
    std::unordered_map<size_t, unsigned int> cache;
    
public:
    unsigned int getVAO(const std::vector<VertexAttribute>& attributes) {
        size_t hash = hashAttributes(attributes);
        
        auto it = cache.find(hash);
        if (it != cache.end()) {
            return it->second;
        }
        
        // Create new VAO
        unsigned int VAO;
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);
        
        for (const auto& attr : attributes) {
            glEnableVertexAttribArray(attr.index);
            glVertexAttribPointer(attr.index, attr.size, attr.type, 
                                attr.normalized, attr.stride, attr.pointer);
        }
        
        cache[hash] = VAO;
        return VAO;
    }
    
private:
    size_t hashAttributes(const std::vector<VertexAttribute>& attributes) {
        // Simple hash function for attributes
        size_t hash = 0;
        for (const auto& attr : attributes) {
            hash ^= std::hash<int>()(attr.index) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};
```

### Batch Rendering:
```cpp
class BatchRenderer {
private:
    struct RenderBatch {
        unsigned int VAO;
        unsigned int texture;
        std::vector<glm::mat4> transforms;
        Shader* shader;
        
        void clear() {
            transforms.clear();
        }
        
        bool canBatch(unsigned int tex, Shader* shdr) {
            return texture == tex && shader == shdr && transforms.size() < MAX_INSTANCES;
        }
    };
    
    static const int MAX_INSTANCES = 1000;
    std::vector<RenderBatch> batches;
    unsigned int instanceVBO;
    
public:
    BatchRenderer() {
        glGenBuffers(1, &instanceVBO);
    }
    
    void submit(unsigned int vao, unsigned int texture, const glm::mat4& transform, Shader* shader) {
        // Try to find existing batch
        for (auto& batch : batches) {
            if (batch.VAO == vao && batch.canBatch(texture, shader)) {
                batch.transforms.push_back(transform);
                return;
            }
        }
        
        // Create new batch
        RenderBatch newBatch;
        newBatch.VAO = vao;
        newBatch.texture = texture;
        newBatch.shader = shader;
        newBatch.transforms.push_back(transform);
        batches.push_back(newBatch);
    }
    
    void flush() {
        for (auto& batch : batches) {
            if (batch.transforms.empty()) continue;
            
            // Upload instance data
            glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
            glBufferData(GL_ARRAY_BUFFER, 
                        batch.transforms.size() * sizeof(glm::mat4),
                        batch.transforms.data(), GL_DYNAMIC_DRAW);
            
            // Setup instanced attributes
            glBindVertexArray(batch.VAO);
            for (int i = 0; i < 4; i++) {
                glEnableVertexAttribArray(3 + i);
                glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, 
                                    sizeof(glm::mat4), (void*)(sizeof(glm::vec4) * i));
                glVertexAttribDivisor(3 + i, 1);
            }
            
            // Render
            batch.shader->use();
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, batch.texture);
            glDrawElementsInstanced(GL_TRIANGLES, indexCount, 
                                  GL_UNSIGNED_INT, 0, batch.transforms.size());
        }
        
        // Clear all batches
        for (auto& batch : batches) {
            batch.clear();
        }
        batches.clear();
    }
};
```

---

## 15. Debugging and Profiling {#debugging}

### OpenGL Debug Output:
```cpp
void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, 
                              GLenum severity, GLsizei length, 
                              const GLchar* message, const void* userParam) {
    std::string sourceStr, typeStr, severityStr;
    
    switch (source) {
        case GL_DEBUG_SOURCE_API:             sourceStr = "API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   sourceStr = "WINDOW SYSTEM"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: sourceStr = "SHADER COMPILER"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     sourceStr = "THIRD PARTY"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     sourceStr = "APPLICATION"; break;
        # Complete OpenGL Expert Guide

## Table of Contents
1. [Introduction to OpenGL](#introduction)
2. [Setting Up Development Environment](#setup)
3. [OpenGL Fundamentals](#fundamentals)
4. [Your First OpenGL Program](#first-program)
5. [Shaders and the Graphics Pipeline](#shaders)
6. [Vertex Data and Buffers](#vertex-data)
7. [Textures and Materials](#textures)
8. [Transformations and Matrices](#transformations)
9. [Lighting and Shading](#lighting)
10. [Advanced Rendering Techniques](#advanced-rendering)
11. [Framebuffers and Post-Processing](#framebuffers)
12. [Geometry and Tessellation Shaders](#geometry-tessellation)
13. [Compute Shaders](#compute-shaders)
14. [Performance Optimization](#optimization)
15. [Debugging and Profiling](#debugging)
16. [Modern OpenGL Best Practices](#best-practices)

---

## 1. Introduction to OpenGL {#introduction}

OpenGL (Open Graphics Library) is a cross-platform, language-independent API for rendering 2D and 3D graphics. It provides a low-level interface to graphics hardware, allowing developers to create high-performance graphics applications.

### Key Concepts:
- **State Machine**: OpenGL is essentially a large state machine
- **Context**: All OpenGL operations happen within a context
- **Pipeline**: Graphics data flows through a programmable pipeline
- **Immediate Mode vs Retained Mode**: Modern OpenGL favors retained mode (storing data on GPU)

### OpenGL Versions:
- **Legacy OpenGL (1.x-2.x)**: Fixed-function pipeline, immediate mode
- **Modern OpenGL (3.x+)**: Programmable pipeline, core profile
- **OpenGL 4.x**: Compute shaders, tessellation, advanced features
- **OpenGL ES**: Mobile/embedded version

### Graphics Pipeline Overview:
```
Vertex Data → Vertex Shader → Tessellation → Geometry Shader → 
Rasterization → Fragment Shader → Per-Sample Operations → Framebuffer
```

---

## 2. Setting Up Development Environment {#setup}

### Required Libraries:

#### Core Libraries:
- **GLFW**: Window management and input handling
- **GLAD/GLEW**: OpenGL extension loading
- **GLM**: Mathematics library for graphics

#### Optional but Recommended:
- **ASSIMP**: 3D model loading
- **SOIL2/stb_image**: Image loading
- **Dear ImGui**: Immediate mode GUI
- **FreeType**: Font rendering

### CMake Setup Example:
```cmake
cmake_minimum_required(VERSION 3.16)
project(OpenGLApp)

set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# Add executable
add_executable(opengl_app
    src/main.cpp
    src/shader.cpp
    src/glad.c  # Include GLAD source
)

# Link libraries
target_link_libraries(opengl_app
    glfw
    ${CMAKE_DL_LIBS}  # For dynamic loading
)

target_include_directories(opengl_app PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/glad/include
)
```

### Basic Project Structure:
```
opengl_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── shader.cpp
│   ├── shader.h
│   └── glad.c
├── shaders/
│   ├── vertex.glsl
│   └── fragment.glsl
├── textures/
├── models/
└── include/
    └── glad/
```

---

## 3. OpenGL Fundamentals {#fundamentals}

### The OpenGL State Machine:
```cpp
// OpenGL is a state machine - you set states that affect subsequent operations
glEnable(GL_DEPTH_TEST);          // Enable depth testing
glDepthFunc(GL_LESS);             // Set depth comparison function
glClearColor(0.2f, 0.3f, 0.3f, 1.0f);  // Set clear color

// These settings remain active until changed
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
```

### OpenGL Objects:
Everything in OpenGL is represented by objects with unique IDs:

```cpp
// Vertex Array Object
GLuint VAO;
glGenVertexArrays(1, &VAO);
glBindVertexArray(VAO);

// Vertex Buffer Object
GLuint VBO;
glGenBuffers(1, &VBO);
glBindBuffer(GL_ARRAY_BUFFER, VBO);

// Texture Object
GLuint texture;
glGenTextures(1, &texture);
glBindTexture(GL_TEXTURE_2D, texture);
```

### Error Handling:
```cpp
GLenum glCheckError_(const char* file, int line) {
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR) {
        std::string error;
        switch (errorCode) {
            case GL_INVALID_ENUM:      error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:     error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: error = "INVALID_OPERATION"; break;
            case GL_OUT_OF_MEMORY:     error = "OUT_OF_MEMORY"; break;
            default:                   error = "UNKNOWN"; break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}

#define glCheckError() glCheckError_(__FILE__, __LINE__)
```

---

## 4. Your First OpenGL Program {#first-program}

### Basic Window Creation with GLFW:
```cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// Settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

int main() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, 
                                         "OpenGL Window", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Input
        processInput(window);

        // Render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
```

### Drawing Your First Triangle:
```cpp
// Vertex data
float vertices[] = {
    -0.5f, -0.5f, 0.0f,  // Bottom left
     0.5f, -0.5f, 0.0f,  // Bottom right
     0.0f,  0.5f, 0.0f   // Top
};

// Vertex shader source
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
)";

// Fragment shader source
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}
)";

// In main function, after OpenGL context creation:

// Build and compile shaders
GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
glCompileShader(vertexShader);

GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
glCompileShader(fragmentShader);

// Link shaders
GLuint shaderProgram = glCreateProgram();
glAttachShader(shaderProgram, vertexShader);
glAttachShader(shaderProgram, fragmentShader);
glLinkProgram(shaderProgram);

// Clean up shaders
glDeleteShader(vertexShader);
glDeleteShader(fragmentShader);

// Set up vertex data and buffers
GLuint VBO, VAO;
glGenVertexArrays(1, &VAO);
glGenBuffers(1, &VBO);

glBindVertexArray(VAO);
glBindBuffer(GL_ARRAY_BUFFER, VBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
glEnableVertexAttribArray(0);

// In render loop:
glUseProgram(shaderProgram);
glBindVertexArray(VAO);
glDrawArrays(GL_TRIANGLES, 0, 3);
```

---

## 5. Shaders and the Graphics Pipeline {#shaders}

### Shader Class Implementation:
```cpp
// shader.h
#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader {
public:
    unsigned int ID;

    Shader(const char* vertexPath, const char* fragmentPath);
    
    void use() const;
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, const glm::vec2& value) const;
    void setVec3(const std::string& name, const glm::vec3& value) const;
    void setVec4(const std::string& name, const glm::vec4& value) const;
    void setMat2(const std::string& name, const glm::mat2& mat) const;
    void setMat3(const std::string& name, const glm::mat3& mat) const;
    void setMat4(const std::string& name, const glm::mat4& mat) const;

private:
    void checkCompileErrors(unsigned int shader, std::string type);
};

// shader.cpp
Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    std::string vertexCode, fragmentCode;
    std::ifstream vShaderFile, fShaderFile;
    
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
    try {
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;
        
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        
        vShaderFile.close();
        fShaderFile.close();
        
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
    }
    catch (std::ifstream::failure& e) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ" << std::endl;
    }
    
    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();
    
    // Compile shaders
    unsigned int vertex, fragment;
    
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");
    
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");
    
    // Shader program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");
    
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Shader::use() const {
    glUseProgram(ID);
}

void Shader::setMat4(const std::string& name, const glm::mat4& mat) const {
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, 
                       glm::value_ptr(mat));
}

void Shader::checkCompileErrors(unsigned int shader, std::string type) {
    int success;
    char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type 
                      << "\n" << infoLog << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type 
                      << "\n" << infoLog << std::endl;
        }
    }
}
```

### Advanced Vertex Shader Example:
```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalMatrix * aNormal;
    TexCoord = aTexCoord;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
```

### Advanced Fragment Shader Example:
```glsl
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform Material material;
uniform Light light;
uniform vec3 viewPos;

void main() {
    // Ambient
    vec3 ambient = light.ambient * texture(material.diffuse, TexCoord).rgb;
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * texture(material.diffuse, TexCoord).rgb;
    
    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * texture(material.specular, TexCoord).rgb;
    
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}
```

---

## 6. Vertex Data and Buffers {#vertex-data}

### Vertex Array Objects (VAO) and Vertex Buffer Objects (VBO):
```cpp
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;
    
    GLuint VAO, VBO, EBO;
    
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, 
         std::vector<Texture> textures) {
        this->vertices = vertices;
        this->indices = indices;
        this->textures = textures;
        
        setupMesh();
    }
    
    void Draw(Shader& shader) {
        // Bind textures
        unsigned int diffuseNr = 1;
        unsigned int specularNr = 1;
        unsigned int normalNr = 1;
        unsigned int heightNr = 1;
        
        for (unsigned int i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            
            std::string number;
            std::string name = textures[i].type;
            if (name == "texture_diffuse")
                number = std::to_string(diffuseNr++);
            else if (name == "texture_specular")
                number = std::to_string(specularNr++);
            else if (name == "texture_normal")
                number = std::to_string(normalNr++);
            else if (name == "texture_height")
                number = std::to_string(heightNr++);
            
            glUniform1i(glGetUniformLocation(shader.ID, 
                       (name + number).c_str()), i);
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }
        
        // Draw mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glActiveTexture(GL_TEXTURE0);
    }

private:
    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        
        glBindVertexArray(VAO);
        
        // Load vertex data
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), 
                     &vertices[0], GL_STATIC_DRAW);
        
        // Load indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                     &indices[0], GL_STATIC_DRAW);
        
        // Vertex positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        
        // Vertex normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
                             (void*)offsetof(Vertex, Normal));
        
        // Vertex texture coordinates
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
                             (void*)offsetof(Vertex, TexCoords));
        
        // Vertex tangent
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
                             (void*)offsetof(Vertex, Tangent));
        
        // Vertex bitangent
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
                             (void*)offsetof(Vertex, Bitangent));
        
        glBindVertexArray(0);
    }
};
```

### Buffer Objects and Memory Management:
```cpp
// Different buffer usage patterns
glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);   // Set once, draw many
glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);  // Change often, draw many
glBufferData(GL_ARRAY_BUFFER, size, data, GL_STREAM_DRAW);   // Change every frame

// Buffer sub-data updates (more efficient than full buffer replacement)
glBufferSubData(GL_ARRAY_BUFFER, offset, size, data);

// Memory mapping for large data transfers
void* ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
// Write data to ptr
glUnmapBuffer(GL_ARRAY_BUFFER);

// Instanced rendering setup
glGenBuffers(1, &instanceVBO);
glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * amount, &modelMatrices[0], 
             GL_STATIC_DRAW);

// Set up instanced vertex attributes
glEnableVertexAttribArray(3);
glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
glEnableVertexAttribArray(4);
glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), 
                     (void*)(sizeof(glm::vec4)));
glEnableVertexAttribArray(5);
glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), 
                     (void*)(2 * sizeof(glm::vec4)));
glEnableVertexAttribArray(6);
glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), 
                     (void*)(3 * sizeof(glm::vec4)));

glVertexAttribDivisor(3, 1);
glVertexAttribDivisor(4, 1);
glVertexAttribDivisor(5, 1);
glVertexAttribDivisor(6, 1);
```

---

## 7. Textures and Materials {#textures}

### Texture Loading and Management:
```cpp
class Texture {
public:
    unsigned int ID;
    std::string type;
    std::string path;
    
    static unsigned int loadTexture(const char* path, bool gammaCorrection = false) {
        unsigned int textureID;
        glGenTextures(1, &textureID);
        
        int width, height, nrComponents;
        unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
        if (data) {
            GLenum format;
            if (nrComponents == 1)
                format = GL_RED;
            else if (nrComponents == 3)
                format = GL_RGB;
            else if (nrComponents == 4)
                format = GL_RGBA;
            
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, 
                        format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            
            stbi_image_free(data);
        } else {
            std::cout << "Texture failed to load at path: " << path << std::endl;
            stbi_image_free(data);
        }
        
        return textureID;
    }
    
    static unsigned int loadCubemap(std::vector<std::string> faces) {
        unsigned int textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
        
        int width, height, nrChannels;
        for (unsigned int i = 0; i < faces.size(); i++) {
            unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, 
                                           &nrChannels, 0);
            if (data) {
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                            0, GL_RGB, width, height, 0, GL_RGB, 
                            GL_UNSIGNED_BYTE, data);
                stbi_image_free(data);
            } else {
                std::cout << "Cubemap failed to load at path: " << faces[i] << std::endl;
                stbi_image_free(data);
            }
        }
        
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        
        return textureID;
    }
};

// Advanced texture features
void setupAdvancedTexture(unsigned int& textureID) {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    // Anisotropic filtering
    if (GLAD_GL_EXT_texture_filter_anisotropic) {
        float maxAnisotropy;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAnisotropy);
    }
    
    // Texture arrays for batch processing
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, width, height, layerCount, 
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}
```

### Material System:
```cpp
struct Material {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    
    unsigned int diffuseMap;
    unsigned int specularMap;
    unsigned int normalMap;
    unsigned int heightMap;
    unsigned int roughnessMap;
    unsigned int metallicMap;
    
    void bind(Shader& shader) {
        shader.setVec3("material.ambient", ambient);
        shader.setVec3("material.diffuse", diffuse);
        shader.setVec3("material.specular", specular);
        shader.setFloat("material.shininess", shininess);
        
        // Bind texture maps
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseMap);
        shader.setInt("material.diffuseMap", 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, specularMap);
        shader.setInt("material.specularMap", 1);
        
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, normalMap);
        shader.setInt("material.normalMap", 2);
    }
};
```

---

## 8. Transformations and Matrices {#transformations}

### Transformation Hierarchy:
```cpp
class Transform {
public:
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};  // Euler angles in degrees
    glm::vec3 scale{1.0f};
    
    glm::mat4 getModelMatrix() const {
        glm::mat4 translation = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 rotationX = glm::rotate(glm::mat4(1.0f), 
                                         glm::radians(rotation.x), glm::vec3(1, 0, 0));
        glm::mat4 rotationY = glm::rotate(glm::mat4(1.0f), 
                                         glm::radians(rotation.y), glm::vec3(0, 1, 0));
        glm::mat4 rotationZ = glm::rotate(glm::mat4(1.0f), 
                                         glm::radians(rotation.z), glm::vec3(0, 0, 1));
        glm::mat4 rotationMatrix = rotationZ * rotationY * rotationX;
        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
        
        return translation * rotationMatrix * scaleMatrix;
    }
    
    glm::mat3 getNormalMatrix() const {
        return glm::mat3(glm::transpose(glm::inverse(getModelMatrix())));
    }
};

class Camera {
public:
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    
    float Yaw;
    float Pitch;
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;
    
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), 
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), 
           float yaw = -90.0f, float pitch = 0.0f) 
        : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(2.5f), 
          MouseSensitivity(0.1f), Zoom(45.0f) {
        Position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }
    
    glm::mat4 GetViewMatrix() {
        return glm::lookAt(Position, Position + Front, Up);
    }
    
    glm::mat4 GetProjectionMatrix(float aspectRatio, float near = 0.1f, float far = 100.0f) {
        return glm::perspective(glm::radians(Zoom), aspectRatio, near, far);
    }
    
    void ProcessKeyboard(Camera_Movement direction, float deltaTime) {
        float velocity = MovementSpeed * deltaTime;
        if (direction == FORWARD)
            Position += Front * velocity;
        if (direction == BACKWARD)
            Position -= Front * velocity;
        if (direction == LEFT)
            Position -= Right * velocity;
        if (direction == RIGHT)
            Position += Right * velocity;
    }
    
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true) {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;
        
        Yaw += xoffset;
        Pitch += yoffset;
        
        if (constrainPitch) {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }
        
        updateCameraVectors();
    }
    
private:
    void updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);
        
        Right = glm::normalize(glm::cross(Front, WorldUp));
        Up = glm::normalize(glm::cross(Right, Front));
    }
};

// Quaternion-based rotations (more stable)
class QuaternionTransform {
public:
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};  // w, x, y, z
    glm::vec3 scale{1.0f};
    
    glm::mat4 getModelMatrix() const {
        glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 R = glm::mat4_cast(rotation);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        return T * R * S;
    }
    
    void rotate(float angle, const glm::vec3& axis) {
        glm::quat deltaRotation = glm::angleAxis(glm::radians(angle), glm::normalize(axis));
        rotation = deltaRotation * rotation;
    }
    
    void lookAt(const glm::vec3& target, const glm::vec3& up = glm::vec3(0, 1, 0)) {
        glm::vec3 forward = glm::normalize(target - position);
        rotation = glm::quatLookAt(forward, up);
    }
};
```

---

## 9. Lighting and Shading {#lighting}

### Phong Lighting Model Implementation:
```glsl
// Vertex Shader
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalMatrix * aNormal;
    TexCoords = aTexCoords;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}

// Fragment Shader
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    sampler2D normal;
    float shininess;
};

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {
    vec3 position;
    
    float constant;
    float linear;
    float quadratic;
    
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    
    float constant;
    float linear;
    float quadratic;
    
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

#define NR_POINT_LIGHTS 4

uniform vec3 viewPos;
uniform DirLight dirLight;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight spotLight;
uniform Material material;

// Function prototypes
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir);
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

void main() {
    // Properties
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Phase 1: Directional lighting
    vec3 result = CalcDirLight(dirLight, norm, viewDir);
    
    // Phase 2: Point lights
    for(int i = 0; i < NR_POINT_LIGHTS; i++)
        result += CalcPointLight(pointLights[i], norm, FragPos, viewDir);
    
    // Phase 3: Spot light
    result += CalcSpotLight(spotLight, norm, FragPos, viewDir);
    
    FragColor = vec4(result, 1.0);
}

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    
    // Diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    
    // Combine results
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));
    
    return (ambient + diffuse + specular);
}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    
    // Attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                              light.quadratic * (distance * distance));
    
    // Combine results
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));
    
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    
    return (ambient + diffuse + specular);
}
```

### Physically Based Rendering (PBR):
```glsl
// PBR Fragment Shader
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

// Material parameters
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

// Lights
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 camPos;

const float PI = 3.14159265359;

// Easy trick to get tangent-normals to world-space to keep PBR code simplified
vec3 getNormalFromMap() {
    vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;

    vec3 Q1  = dFdx(WorldPos);
    vec3 Q2  = dFdy(WorldPos);
    vec2 st1 = dFdx(TexCoords);
    vec2 st2 = dFdy(TexCoords);

    vec3 N   = normalize(Normal);
    vec3 T  = normalize(Q1 * st2.t - Q2 * st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {		
    vec3 albedo = pow(texture(albedoMap, TexCoords).rgb, 2.2);
    float metallic = texture(metallicMap, TexCoords).r;
    float roughness = texture(roughnessMap, TexCoords).r;
    float ao = texture(aoMap, TexCoords).r;

    vec3 N = getNormalFromMap();
    vec3 V = normalize(camPos - WorldPos);

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);
	           
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) {
        vec3 L = normalize(lightPositions[i] - WorldPos);
        vec3 H = normalize(V + L);
        float distance    = length(lightPositions[i] - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance     = lightColors[i] * attenuation;        
        
        float NDF = DistributionGGX(N, H, roughness);        
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;	  
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular     = numerator / denominator;  
            
        float NdotL = max(dot(N, L), 0.0);        
        Lo += (kD * albedo / PI + specular) * radiance * NdotL; 
    }   
  
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;
	
    color = color / (color + vec3(1.0));
    color = pow(color, vec1/2.2);  
   
    FragColor = vec4(color, 1.0);
}
```

---

## 10. Advanced Rendering Techniques {#advanced-rendering}

### Shadow Mapping:
```cpp
// Shadow mapping setup
class ShadowMap {
private:
    unsigned int depthMapFBO;
    unsigned int depthMap;
    const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;

public:
    ShadowMap() {
        glGenFramebuffers(1, &depthMapFBO);
        
        glGenTextures(1, &depthMap);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
                     SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    void beginShadowPass() {
        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
    }
    
    void endShadowPass(int screenWidth, int screenHeight) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, screenWidth, screenHeight);
    }
    
    unsigned int getDepthMap() { return depthMap; }
};

// Shadow mapping fragment shader
const char* shadowFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec4 FragPosLightSpace;
} fs_in;

uniform sampler2D diffuseTexture;
uniform sampler2D shadowMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

float ShadowCalculation(vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    
    float bias = 0.005;
    float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    
    return shadow;
}

void main() {           
    vec3 color = texture(diffuseTexture, fs_in.TexCoords).rgb;
    vec3 normal = normalize(fs_in.Normal);
    vec3 lightColor = vec3(0.15);
    
    // Ambient
    vec3 ambient = 0.15 * color;
    
    // Diffuse
    vec3 lightDir = normalize(lightPos - fs_in.FragPos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = 0.0;
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 specular = spec * lightColor;    
    
    // Calculate shadow
    float shadow = ShadowCalculation(fs_in.FragPosLightSpace);                      
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;    
    
    FragColor = vec4(lighting, 1.0);
}
)";
```

### Screen Space Ambient Occlusion (SSAO):
```cpp
class SSAO {
private:
    unsigned int ssaoFBO, ssaoBlurFBO;
    unsigned int ssaoColorBuffer, ssaoColorBufferBlur;
    unsigned int noiseTexture;
    std::vector<glm::vec3> ssaoKernel;
    std::vector<glm::vec3> ssaoNoise;
    
public:
    SSAO(int screenWidth, int screenHeight) {
        // Generate sample kernel
        std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0);
        std::default_random_engine generator;
        
        for (unsigned int i = 0; i < 64; ++i) {
            glm::vec3 sample(randomFloats(generator) * 2.0 - 1.0, 
                           randomFloats(generator) * 2.0 - 1.0, 
                           randomFloats(generator));
            sample = glm::normalize(sample);
            sample *= randomFloats(generator);
            
            float scale = float(i) / 64.0f;
            scale = lerp(0.1f, 1.0f, scale * scale);
            sample *= scale;
            ssaoKernel.push_back(sample);
        }
        
        // Generate noise texture
        for (unsigned int i = 0; i < 16; i++) {
            glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, 
                          randomFloats(generator) * 2.0 - 1.0, 
                          0.0f);
            ssaoNoise.push_back(noise);
        }
        
        glGenTextures(1, &noiseTexture);
        glBindTexture(GL_TEXTURE_2D, noiseTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
        // Setup framebuffers...
    }
    
    float lerp(float a, float b, float f) {
        return a + f * (b - a);
    }
};

// SSAO fragment shader
const char* ssaoFragmentShader = R"(
#version 330 core
out float FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D texNoise;

uniform vec3 samples[64];
uniform mat4 projection;

int kernelSize = 64;
float radius = 0.5;
float bias = 0.025;

const vec2 noiseScale = vec2(800.0/4.0, 600.0/4.0); 

void main() {
    vec3 fragPos = texture(gPosition, TexCoords).xyz;
    vec3 normal = normalize(texture(gNormal, TexCoords).rgb);
    vec3 randomVec = normalize(texture(texNoise, TexCoords * noiseScale).xyz);
    
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    
    float occlusion = 0.0;
    for(int i = 0; i < kernelSize; ++i) {
        vec3 samplePos = TBN * samples[i];
        samplePos = fragPos + samplePos * radius; 
        
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;
        
        float sampleDepth = texture(gPosition, offset.xy).z;
        
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;           
    }
    occlusion = 1.0 - (occlusion / kernelSize);
    
    FragColor = occlusion;
}
)";
```

### Instanced Rendering:
```cpp
// Instanced rendering for thousands of objects
void setupInstancedRendering() {
    // Generate model matrices for instances
    std::vector<glm::mat4> modelMatrices;
    int amount = 10000;
    float radius = 50.0;
    float offset = 2.5f;
    
    for (int i = 0; i < amount; i++) {
        glm::mat4 model = glm::mat4(1.0f);
        
        float angle = (float)i / (float)amount * 360.0f;
        float displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
        float x = sin(angle) * radius + displacement;
        displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
        float y = displacement * 0.4f;
        displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
        float z = cos(angle) * radius + displacement;
        model = glm::translate(model, glm::vec3(x, y, z));
        
        float scale = (rand() % 20) / 100.0f + 0.05;
        model = glm::scale(model, glm::vec3(scale));
        
        float rotAngle = (rand() % 360);
        model = glm::rotate(model, rotAngle, glm::vec3(0.4f, 0.6f, 0.8f));
        
        modelMatrices.push_back(model);
    }
    
    // Store in buffer
    unsigned int buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, amount * sizeof(glm::mat4), &modelMatrices[0], GL_STATIC_DRAW);
    
    // Set up instanced vertex attributes
    for (unsigned int i = 0; i < meshes.size(); i++) {
        unsigned int VAO = meshes[i].VAO;
        glBindVertexArray(VAO);
        
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4)));
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));

        glVertexAttribDivisor(3, 1);
        glVertexAttribDivisor(4, 1);
        glVertexAttribDivisor(5, 1);
        glVertexAttribDivisor(6, 1);

        glBindVertexArray(0);
    }
}

// Instanced vertex shader
const char* instancedVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in mat4 aInstanceMatrix;

out vec2 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main() {
    TexCoords = aTexCoords;
    gl_Position = projection * view * aInstanceMatrix * vec4(aPos, 1.0f);
}
)";

// Draw instanced
glDrawElementsInstanced(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0, amount);
```

---

## 11. Framebuffers and Post-Processing {#framebuffers}

### Framebuffer Setup:
```cpp
class Framebuffer {
private:
    unsigned int FBO;
    unsigned int colorTexture;
    unsigned int depthTexture;
    int width, height;
    
public:
    Framebuffer(int w, int h) : width(w), height(h) {
        // Generate framebuffer
        glGenFramebuffers(1, &FBO);
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        
        // Color attachment
        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        