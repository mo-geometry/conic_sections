# Session 1 Homework — GPU Pipeline Foundations

Estimated time: 60–90 minutes

## Part 1: Understand What You Built (20 min)

Read through these files in order. For each one, write a one-sentence
comment above every function or class explaining what it does in your
own words. Don't copy the docstrings — restate them as if explaining
to a colleague.

1. `src/conic_sections/core/transforms.py`
2. `src/conic_sections/core/orbit_camera.py`
3. `src/conic_sections/assets/mesh.py`
4. `src/conic_sections/__main__.py`

Questions to answer in a notebook (physical or digital):

- In `look_at()`, why do we compute `right = cross(forward, up)` and
  then `true_up = cross(right, forward)`? Why not just use the `up`
  vector directly?

- In `OrbitCamera`, the elevation is clamped to ±(π/2 - 0.01). What
  would happen visually if we allowed exactly ±π/2?

- In `__main__.py`, why do we call `.T.copy()` on matrices before
  uploading them to the GPU? What would happen if we sent them without
  transposing?

## Part 2: The OpenGL Pipeline (20 min)

Read this single resource — it's the clearest explanation of how
vertices become pixels:

**"LearnOpenGL — Hello Triangle"**
https://learnopengl.com/Getting-started/Hello-Triangle

Focus on understanding these concepts (you don't need to write C++):

- What is a **Vertex Array Object (VAO)** and why does it exist?
- What does the **vertex shader** do vs the **fragment shader**?
- What is the difference between **model**, **view**, and **projection**
  matrices? How do they compose?
- What does "rasterisation" mean — what happens between the vertex
  shader and the fragment shader?

Then read:
**"LearnOpenGL — Coordinate Systems"**
https://learnopengl.com/Getting-started/Coordinate-Systems

This explains the full transform chain: local space → world space →
view space → clip space → screen space. Map each stage to the code
in our `__main__.py`.

## Part 3: Row-Major vs Column-Major (10 min)

This is the single most common source of bugs in graphics programming,
and we hit it today. Understand it deeply:

Take a 4x4 translation matrix that moves a point by (3, 5, 7).

Write it out on paper in both conventions:
- **Row-major** (how numpy stores it, how you'd write it in a textbook)
- **Column-major** (how OpenGL reads it from memory)

Then answer:
- If you write the 16 floats of the row-major matrix into a flat array
  left-to-right, top-to-bottom, and hand that to OpenGL — what does
  OpenGL think the translation is?
- Why does `.T.copy().tobytes()` fix this?

## Part 4: Shaders — Read and Annotate (10 min)

Open `src/conic_sections/rendering/shaders.py`.

For the fragment shader, annotate each section:
- What does `normalize(frag_normal)` do and why is it needed?
- What is the `halfway` vector in Blinn-Phong, and how does it differ
  from the reflection vector in standard Phong?
- What does `pow(max(dot(norm, halfway), 0.0), 32.0)` control? What
  happens if you change 32.0 to 4.0? To 256.0?

**Experiment**: change the specular exponent in the shader and re-run
the application. Observe the difference. Try values 4, 32, 128, 256.

## Part 5: Arm GPU Architecture (15 min)

Read this Arm blog post on the Mali GPU architecture:

**"The Arm Mali GPU — An Abstract Machine"**
https://developer.arm.com/documentation/102849/latest/

Key concepts to understand:
- **Tile-based rendering** — how does Mali differ from desktop GPUs
  like NVIDIA? Why does this matter for mobile/embedded?
- **Early-Z rejection** — what is it and why does draw order matter?
- **Shader core architecture** — how do Mali shader cores differ from
  NVIDIA CUDA cores?

Write down three things that surprise you or that you didn't know.

## Stretch Goal: Project Structure

If you have extra time, read the Python Packaging User Guide section
on src layout:

https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/

Answer: why do we use `src/conic_sections/` instead of just
`conic_sections/` at the project root? What problem does the extra
directory level solve?

---

## Checklist

- [ ] Annotated transforms.py, orbit_camera.py, mesh.py, __main__.py
- [ ] Answered the three code questions
- [ ] Read LearnOpenGL Hello Triangle + Coordinate Systems
- [ ] Wrote out row-major vs column-major translation matrix
- [ ] Annotated the fragment shader
- [ ] Experimented with specular exponent values
- [ ] Read the Arm Mali GPU architecture post
- [ ] Wrote down three surprises from the Arm reading
