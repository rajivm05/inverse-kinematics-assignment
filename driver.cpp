// A driver to perform inverse kinematics with skinning.

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

#include "sceneObjectDeformable.h"
#include "lighting.h"
#include "cameraLighting.h"
#include "openGL-headers.h"
#include "camera.h"
#include "objMesh.h"
#include "performanceCounter.h"
#include "averagingBuffer.h"
#include "inputDevice.h"
#include "openGLHelper.h"
#include "valueIndex.h"
#include "configFile.h"
#include "skinning.h"
#include "FK.h"
#include "IK.h"
#include "handleControl.h"
#include "skeletonRenderer.h"
#ifdef WIN32
  #include <windows.h>
#endif
#include <vector>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <climits>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

#include <adolc/adolc.h>
using namespace std;

static string meshFilename;
static string configFilename;
static string screenshotBaseName;
static string jointHierarchyFilename;
static string jointWeightsFilename;
static string jointRestTransformsFilename;

static bool fullScreen = 0;
static bool showAxes = false;
static bool showWireframe = true;
static bool showObject = true;
static bool showBoneColors = false;
static bool useLighting = true;
static double allLightsIntensity = 1.0;

static Vec3d modelCenter(0.0);
static double modelRadius = 1.0;
static ObjMesh * mesh = nullptr;
static SceneObjectDeformable * meshDeformable = nullptr;

static FK * fk = nullptr;
static IK * ik = nullptr;
static IK * ikJoint = nullptr;   // joint-handle IK (preloaded if IKJointIDs is set)
static IK * ikVertex = nullptr;  // vertex-handle IK (preloaded if IKVertexIDs is set)
static Skinning * skinning = nullptr;
static SkeletonRenderer * skeletonRenderer = nullptr;

static bool renderSkeleton = true;
static int curJointID = -1;

static SphericalCamera * camera = nullptr;
static int windowWidth = 800, windowHeight = 600;
static double zNear = 0.001, zFar = 1000;
static int selectedVertex = -1;

static int windowID = 0;
static int graphicsFrameID = 0;

static Lighting * lighting = nullptr;
static CameraLighting * cameraLighting = nullptr;

static InputDevice id;
static bool reverseHandle = false;
static HandleControl handleControl;

static PerformanceCounter counter, titleBarCounter;
static int titleBarFrameCounter = 0;
static AveragingBuffer fpsBuffer(5);

static vector<int> IKJointIDs;
static vector<int> IKVertexIDs;      // optional: when set, IK handles are mesh vertices
static bool useVertexHandles = false;
static vector<Vec3d> IKJointPos;     // stores handle positions in either mode

// --- Pose recording / playback ---
// Snapshot full joint Euler angles with 'p'; toggle loop playback with space; clear with 'x'.
static vector<vector<Vec3d>> recordedPoses;
static bool isPlaying = false;
static double playbackT = 0.0;        // progress in current segment, [0,1)
static int    playbackSeg = 0;        // current segment index (seg i: pose i -> pose i+1)
static double secondsPerSegment = 1.0;

// --- Frame capture (toggle with 'F') ---
// Writes PPM (P6) frames. Convert to JPEG with:
//   ffmpeg -framerate 15 -i frames/frame_%04d.ppm -q:v 2 frames/frame_%04d.jpg
// or on macOS:
//   for f in frames/*.ppm; do sips -s format jpeg "$f" --out "${f%.ppm}.jpg"; done
static bool   captureFrames = false;
static int    captureFrameIndex = 0;
static string captureFolder = "frames";

//======================= Functions =============================

// h,s,v in [0,1] -> rgb in [0,1].
static Vec3d hsv2rgb(double h, double s, double v)
{
  double H = std::fmod(h, 1.0) * 6.0;
  int i = (int)std::floor(H);
  double f = H - i;
  double p = v * (1.0 - s);
  double q = v * (1.0 - s * f);
  double t = v * (1.0 - s * (1.0 - f));
  switch (((i % 6) + 6) % 6)
  {
    case 0: return Vec3d(v, t, p);
    case 1: return Vec3d(q, v, p);
    case 2: return Vec3d(p, v, t);
    case 3: return Vec3d(p, q, v);
    case 4: return Vec3d(t, p, v);
    default: return Vec3d(v, p, q);
  }
}

// Build per-vertex colors by blending a distinct color per joint with the skinning weights.
// Gives each bone a visibly different hue and paints its influenced verts that color.
static void computeVertexBoneColors()
{
  const int numJoints = fk->getNumJoints();
  const int nV = skinning->getNumMeshVertices();
  const int K  = skinning->getNumJointsInfluencingEachVertex();
  const int * jointIdx = skinning->getMeshSkinningJoints();
  const double * weights = skinning->getMeshSkinningWeights();

  // Golden-ratio hue spacing -> maximally separated colors regardless of numJoints.
  const double phi = 0.61803398875;
  std::vector<Vec3d> palette;
  palette.reserve(numJoints);
  for (int j = 0; j < numJoints; j++)
    palette.push_back(hsv2rgb(std::fmod(j * phi, 1.0), 0.75, 0.95));

  std::vector<Vec3d> colors(nV, Vec3d(0.0, 0.0, 0.0));
  for (int v = 0; v < nV; v++)
  {
    for (int k = 0; k < K; k++)
    {
      int j = jointIdx[v * K + k];
      double w = weights[v * K + k];
      if (j >= 0 && j < numJoints)
        colors[v] += w * palette[j];
    }
  }
  meshDeformable->SetCustomColor(colors);
}

static void ensureCaptureDir()
{
#ifdef WIN32
  _mkdir(captureFolder.c_str());
#else
  mkdir(captureFolder.c_str(), 0755); // harmless if it already exists
#endif
}

// Grab the current back buffer and write it as a PPM (P6) frame.
static void captureCurrentFrame()
{
  int w = windowWidth, h = windowHeight;
  vector<unsigned char> row(3 * w * h);
  glReadBuffer(GL_BACK);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, row.data());

  // GL origin is bottom-left; PPM is top-left. Flip rows.
  vector<unsigned char> flipped(3 * w * h);
  for (int y = 0; y < h; y++)
    memcpy(&flipped[3 * w * (h - 1 - y)], &row[3 * w * y], 3 * w);

  char name[512];
  snprintf(name, sizeof(name), "%s/frame_%04d.ppm", captureFolder.c_str(), captureFrameIndex++);
  FILE * f = fopen(name, "wb");
  if (!f) { perror(name); return; }
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  fwrite(flipped.data(), 1, flipped.size(), f);
  fclose(f);
}

static void snapshotPose()
{
  int n = fk->getNumJoints();
  vector<Vec3d> p(n);
  for (int i = 0; i < n; i++) p[i] = fk->jointEulerAngle(i);
  recordedPoses.push_back(std::move(p));
  cout << "Recorded pose #" << recordedPoses.size() << endl;
}

static void clearRecordedPoses()
{
  recordedPoses.clear();
  isPlaying = false;
  playbackT = 0.0;
  playbackSeg = 0;
  cout << "Cleared all recorded poses." << endl;
}

// Smoothstep-interpolate between poses a and b by parameter t in [0,1].
static void applyPoseLerp(const vector<Vec3d> & a, const vector<Vec3d> & b, double t)
{
  double s = 0.5 - 0.5 * std::cos(M_PI * t);
  int n = fk->getNumJoints();
  for (int i = 0; i < n; i++)
    fk->jointEulerAngle(i) = a[i] + (b[i] - a[i]) * s;
}

// Advance recordedPose playback by dt seconds. Loops back to pose 0.
static void advancePlayback(double dt)
{
  if (recordedPoses.empty()) return;
  if (recordedPoses.size() == 1)
  {
    applyPoseLerp(recordedPoses[0], recordedPoses[0], 0.0);
    return;
  }
  playbackT += dt / secondsPerSegment;
  while (playbackT >= 1.0)
  {
    playbackT -= 1.0;
    playbackSeg = (playbackSeg + 1) % (int)recordedPoses.size();
  }
  int next = (playbackSeg + 1) % (int)recordedPoses.size();
  applyPoseLerp(recordedPoses[playbackSeg], recordedPoses[next], playbackT);
}

static void updateSkinnedMesh()
{
  vector<Vec3d> newPos(meshDeformable->GetNumVertices());
  double * newPosv = (double*)newPos.data();

  fk->computeJointTransforms();

  skinning->applySkinning(fk->getJointSkinTransforms(), newPosv);
  for(size_t i = 0; i < mesh->getNumVertices(); i++)
    mesh->setPosition(i, newPos[i]);

  meshDeformable->BuildNormals();
}

static void switchIKMode(bool toVertex)
{
  if (toVertex && ikVertex == nullptr)
  {
    cout << "Vertex-handle IK not available: IKVertexIDs was empty in the config." << endl;
    return;
  }
  if (!toVertex && ikJoint == nullptr)
  {
    cout << "Joint-handle IK not available: IKJointIDs was empty in the config." << endl;
    return;
  }
  useVertexHandles = toVertex;
  ik = toVertex ? ikVertex : ikJoint;

  size_t numHandles = useVertexHandles ? IKVertexIDs.size() : IKJointIDs.size();
  IKJointPos.resize(numHandles);
  // Refresh target positions from current skinned mesh / joint globals.
  for (size_t i = 0; i < numHandles; i++)
    IKJointPos[i] = (useVertexHandles ? mesh->getPosition(IKVertexIDs[i])
                                      : fk->getJointGlobalPosition(IKJointIDs[i]));

  handleControl.clearHandleSelection();
  cout << "IK handles: " << (useVertexHandles ? "mesh VERTICES" : "skeleton JOINTS")
       << " (" << numHandles << ")" << endl;
}

static Vec3d getHandleWorldPosition(size_t handleIndex)
{
  if (useVertexHandles)
  {
    int vtx = IKVertexIDs[handleIndex];
    return mesh->getPosition(vtx);
  }
  return fk->getJointGlobalPosition(IKJointIDs[handleIndex]);
}

static void resetSkinningToRest()
{
  fk->resetToRestPose();
  updateSkinnedMesh();
  for(size_t i = 0; i < IKJointPos.size(); i++)
  {
    IKJointPos[i] = getHandleWorldPosition(i);
  }
  handleControl.clearHandleSelection();
  curJointID = -1;

  cout << "reset mesh to rest" << endl;
}

static void idleFunction()
{
  glutSetWindow(windowID);
  counter.StopCounter();
  double dt = counter.GetElapsedTime();
  counter.StartCounter();

  if (isPlaying && !recordedPoses.empty())
  {
    // Playback mode: drive joint angles from keyframes, skip drag/IK.
    advancePlayback(dt);
    updateSkinnedMesh();
    // Keep IK handle positions aligned with the current pose so dragging resumes cleanly
    // when the user stops playback.
    for (size_t i = 0; i < IKJointPos.size(); i++)
      IKJointPos[i] = getHandleWorldPosition((int)i);
  }
  else
  {
    // Take appropriate action in case the user is dragging a vertex.
    auto processDrag = [&](int vertex, Vec3d posDiff)
    {
      if (len2(posDiff) > 0 && handleControl.isHandleSelected())
      {
        IKJointPos[handleControl.getSelectedHandle()] += posDiff;
      }
    };
    handleControl.processHandleMovement(id.getMousePosX(), id.getMousePosY(), id.shiftPressed(), processDrag);

    const int maxIKIters = 10;
    const double maxOneStepDistance = modelRadius / 1000;

    ik->doIK(IKJointPos.data(), fk->getJointEulerAngles());

    updateSkinnedMesh();
  }

  titleBarFrameCounter++;
  // update title bar at 4 Hz
  titleBarCounter.StopCounter();
  double elapsedTime = titleBarCounter.GetElapsedTime();
  if (elapsedTime >= 1.0 / 4)
  {
    titleBarCounter.StartCounter();
    double fps = titleBarFrameCounter / elapsedTime;
    fpsBuffer.addValue(fps);

    // update menu bar
    char windowTitle[4096];
    sprintf(windowTitle, "Vertices: %d | %.1f FPS | graphicsFrame %d ", meshDeformable->Getn(), fpsBuffer.getAverage(), graphicsFrameID);
    glutSetWindowTitle(windowTitle);
    titleBarFrameCounter = 0;
  }
  graphicsFrameID++;
  glutPostRedisplay();
}

static void reshape(int x, int y)
{
  glViewport(0,0,x,y);
  glMatrixMode(GL_PROJECTION); 
  glLoadIdentity(); 
  windowWidth = x;
  windowHeight = y;

  // Calculate the aspect ratio of the window
  gluPerspective(45.0f, 1.0 * windowWidth / windowHeight, zNear, zFar);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

static void displayFunction()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  camera->Look(); // calls gluLookAt

  glDisable(GL_LIGHTING);

  glLineWidth(1.0);
  if (showAxes)
    RenderAxes(1);

  if (useLighting)
  {
    glEnable(GL_LIGHTING);
    if (cameraLighting)
      cameraLighting->LightScene(camera);
    else if (lighting)
      lighting->LightScene();
  }
  else
    glDisable(GL_LIGHTING);

  glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE); //only when stencil pass and z-buffer pass, set stencil value to stencil reference
  glStencilFunc(GL_ALWAYS, 1, ~(0u));        //always pass stencil test, stencil renference value is 1

  if(true)
  {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

//    glEnable(GL_POLYGON_OFFSET_FILL);
//    glPolygonOffset(1.0, 1.0);
//    glDrawBuffer(GL_NONE);

    /***********************************
     *    render transparent object
     ***********************************/
//    glDisable(GL_POLYGON_OFFSET_FILL);
//    glDrawBuffer(GL_BACK);
    glDisable(GL_BLEND);
  }

  glColor3f(0.9,0.9,0.9);
  if (showObject)
  {
    meshDeformable->Render();
  }
  glColor3f(0,0,0);
  if (showWireframe)
    meshDeformable->RenderEdges();

  glDisable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glPointSize(10.0); 

  // ---------------------------------------------------------
  //    now rendering non-material/non-texture stuff here
  if (selectedVertex >= 0)
  {
    glColor3f(1,0,0);
    glBegin(GL_POINTS);
    Draw(mesh->getPosition(selectedVertex));
    glEnd();
  }

  glDisable(GL_DEPTH_TEST);
  if (renderSkeleton)
  {
    skeletonRenderer->renderSkeleton();
  }
  if (curJointID >= 0 && curJointID < fk->getNumJoints())
  {
    skeletonRenderer->renderJoint(curJointID);
  }
  if (!useVertexHandles)
  {
    for(int jointID : IKJointIDs)
    {
      skeletonRenderer->renderJointCoordAxes(jointID);
    }
  }
  else
  {
    // Mark each vertex handle with a small red point so the user can see where they are.
    glColor3f(0.9f, 0.2f, 0.2f);
    glPointSize(8.0f);
    glBegin(GL_POINTS);
    for(int vtx : IKVertexIDs)
      Draw(mesh->getPosition(vtx));
    glEnd();
  }
  glEnable(GL_DEPTH_TEST);

  glStencilFunc(GL_ALWAYS, 0, ~(0u)); // always pass stencil test, stencil renference value is set to 0
  // render the vertex currently being manipulated via IK
  if (handleControl.isHandleSelected())
  {
    int handleID = handleControl.getSelectedHandle();
    Vec3d handlePos = getHandleWorldPosition(handleID);
    glColor3f(1,0,0);
    glPointSize(8.0);
    Draw(handlePos);

    // render the moving handle at location IKJointPos[handleID]
    handleControl.renderHandle(camera, IKJointPos[handleID], reverseHandle);
  }

  if (captureFrames)
    captureCurrentFrame();

  glutSwapBuffers();
}

static void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 27:
      exit(0);
    break;

    case '0':
    case 'r':
      resetSkinningToRest();
      camera->Reset();
      break;

    case 9:
      fullScreen = 1 - fullScreen;
      if (fullScreen == 1)
        glutFullScreen();
      else {
        glutReshapeWindow(800, 600);
        glutPositionWindow(5, 150);
      }
      break;

    case '\\':
      camera->Reset();
    break;

    case '=':
      curJointID++;
      if (curJointID >= fk->getNumJoints())
        curJointID = -1;
    break;

    case 'a':
      showAxes = !showAxes;
      break;

    case 'w':
      showWireframe = !showWireframe;
      break;

    case 'e':
      showObject = !showObject;
      break;

    case 's':
      renderSkeleton = !renderSkeleton;
      break;

    case 'd':
    {
      Skinning::SkinningMode mode = skinning->getSkinningMode();
      mode = (mode == Skinning::LINEAR_BLEND) ? Skinning::DUAL_QUATERNION : Skinning::LINEAR_BLEND;
      skinning->setSkinningMode(mode);
      cout << "Skinning mode: " << (mode == Skinning::LINEAR_BLEND ? "Linear Blend (LBS)" : "Dual Quaternion (DQS)") << endl;
      break;
    }

    case 'v':
      switchIKMode(!useVertexHandles);
      break;

    case 'i':
    {
      bool on = !ik->isSubSteppingEnabled();
      if (ikJoint)  ikJoint->setSubSteppingEnabled(on);
      if (ikVertex) ikVertex->setSubSteppingEnabled(on);
      cout << "IK sub-stepping: " << (on ? "ON" : "OFF") << endl;
      break;
    }

    case 'c':
      showBoneColors = !showBoneColors;
      if (showBoneColors) meshDeformable->EnableCustomColor();
      else                meshDeformable->DisableCustomColor();
      cout << "Bone colors: " << (showBoneColors ? "ON" : "OFF") << endl;
      break;

    case 'p':
      snapshotPose();
      break;

    case ' ':
      if (recordedPoses.empty())
      {
        cout << "No poses recorded. Press 'p' to snapshot the current pose." << endl;
        break;
      }
      isPlaying = !isPlaying;
      if (isPlaying) { playbackSeg = 0; playbackT = 0.0; }
      cout << "Playback: " << (isPlaying ? "PLAY" : "PAUSE")
           << " (" << recordedPoses.size() << " poses)" << endl;
      break;

    case 'x':
      clearRecordedPoses();
      break;

    case 'F':
      captureFrames = !captureFrames;
      if (captureFrames)
      {
        ensureCaptureDir();
        captureFrameIndex = 0;
        cout << "Frame capture ON -> writing PPM to '" << captureFolder << "/'" << endl;
      }
      else
      {
        cout << "Frame capture OFF (" << captureFrameIndex << " frames saved)" << endl;
      }
      break;

    default:
      break;
  }
}

static void specialKeysFunc(int key, int x, int y)
{
  switch (key)
  {
    case GLUT_KEY_LEFT:
      camera->MoveFocusRight(0.1 * fabs(camera->GetRadius()));
      break;

    case GLUT_KEY_RIGHT:
      camera->MoveFocusRight(-0.1 * fabs(camera->GetRadius()));
      break;

    case GLUT_KEY_DOWN:
      camera->MoveFocusUp(0.1 * fabs(camera->GetRadius()));
      break;

    case GLUT_KEY_UP:
      camera->MoveFocusUp(-0.1 * fabs(camera->GetRadius()));
      break;
  }
}

static void mouseNoDrag(int x, int y)
{
  id.setMousePos(x,y);
  if (handleControl.isHandleSelected())
  {
    Vec3d worldPos(0.0);
    GLubyte stencilValue;
    float zValue;
    unprojectPointFromScreen(x,y, &worldPos[0], &stencilValue, &zValue);

    if (stencilValue == 1)
    {
      handleControl.setMousePosition(worldPos);
    }
  }
}

static void mouseDrag(int x, int y)
{
  int mouseDeltaX = x-id.getMousePosX();
  int mouseDeltaY = y-id.getMousePosY();

  id.setMousePos(x,y);

  // we moved the camera...
  if (id.rightMouseButtonDown())
  { 
    // right mouse button handles camera rotations
    double scale = 0.2;
    if(id.shiftPressed()) scale *= 0.1;
    camera->MoveRight(scale * mouseDeltaX);
    camera->MoveUp(scale * mouseDeltaY);
  }

  if (id.middleMouseButtonDown() || (id.altPressed() && id.leftMouseButtonDown()))
  { 
    // middle mouse button (or ALT + left mouse button) handles camera translations
    double scale = 0.2 * modelRadius;
    if(id.shiftPressed()) scale *= 0.1;
    camera->ZoomIn(scale * mouseDeltaY);
  }
}

static void mouseButtonActivity(int button, int state, int x, int y)
{
  id.setButton(button, state);
  switch (button)
  {
    case GLUT_LEFT_BUTTON:
    {
      Vec3d clickedPosition(0.0);
      GLubyte stencilValue;
      float zValue = 0.0f;
      unprojectPointFromScreen(x,y, &clickedPosition[0], &stencilValue, &zValue);

      if (id.leftMouseButtonDown())
      {
        if (stencilValue == 0)
        {
          cout << "Clicked on empty space." << endl;
          selectedVertex = -1;
          return;
        }
        MinValueIndex vi;
        for(size_t i = 0; i < mesh->getNumVertices(); i++)
        {
          vi.update(len2(clickedPosition - mesh->getPosition(i)), i);
        }
        selectedVertex = vi.index;
        cout << "Clicked on vertex " << vi.index << endl;

        if (fk->getNumJoints() > 0)
        {
          MinValueIndex vi;
          for(int i = 0; i < fk->getNumJoints(); i++)
          {
            vi.update(len2(fk->getJointGlobalPosition(i) - clickedPosition), i);
          }
          assert(vi.index >= 0);

          if (vi.index != curJointID)
          {
            curJointID = vi.index;
            cout << "select joint ID " << curJointID << ", #joints " << fk->getNumJoints() << endl;
          }
        }
      }

      auto getClosestHandle = [&]() -> int
      {
        MinValueIndex vi;
        size_t numHandles = useVertexHandles ? IKVertexIDs.size() : IKJointIDs.size();
        for(size_t handleID = 0; handleID < numHandles; handleID++)
        {
          vi.update(len2(clickedPosition - getHandleWorldPosition(handleID)), handleID);
        }
        return vi.index;
      };
      auto addOrRemoveHandle = [&]()
      {
        return make_pair(-1, false);
      };
      handleControl.setMouseButtonActivity(id.leftMouseButtonDown(), stencilValue == 1, false,
          clickedPosition, zValue, getClosestHandle, addOrRemoveHandle);

      break;
    }

    case GLUT_MIDDLE_BUTTON:
      break;

    case GLUT_RIGHT_BUTTON:
      break;
  }
}

static void initialize()
{
  // initialize random number generator
  srand(time(nullptr));

  // detect the OpenGL version being used
  printf("GL_VENDOR: %s\n",glGetString(GL_VENDOR));
  printf("GL_RENDERER: %s\n",glGetString(GL_RENDERER));
  printf("GL_VERSION: %s\n",glGetString(GL_VERSION));

  mesh = new ObjMesh(meshFilename);
  meshDeformable = new SceneObjectDeformable(mesh, false);

  if (meshDeformable->HasTextures())
  {
    meshDeformable->EnableTextures();
    meshDeformable->SetUpTextures(SceneObject::MODULATE, SceneObject::NOMIPMAP);
  }
  meshDeformable->BuildNeighboringStructure();
  meshDeformable->BuildNormals();
  //  meshDeformable->BuildDisplayList();

  // ---------------------------------------------------
  // joint initialization
  // ---------------------------------------------------

  if (IKJointIDs.size() == 0 && IKVertexIDs.size() == 0)
  {
    cout << "No IK handles specified in the config file (need IKJointIDs and/or IKVertexIDs)" << endl;
    exit(0);
  }

  assert(jointRestTransformsFilename.size() > 0 && jointWeightsFilename.size() > 0);
  skinning = new Skinning(meshDeformable->Getn(), meshDeformable->GetVertexRestPositions(), jointWeightsFilename);
  fk = new FK(jointHierarchyFilename, jointRestTransformsFilename);

  // Precompute per-vertex colors (static — skinning weights don't change at runtime).
  // Toggle display with the 'c' key.
  computeVertexBoneColors();

  // ---------------------------------------------------
  // Setting up Adol-c (preload whichever modes the config enables; distinct ADOL-C tag per mode)
  // ---------------------------------------------------
  if (IKJointIDs.size() > 0)
    ikJoint = new IK((int)IKJointIDs.size(), IKJointIDs.data(), fk, /*tag*/ 1);
  if (IKVertexIDs.size() > 0)
  {
    cout << "Preloading vertex-handle IK with " << IKVertexIDs.size() << " mesh vertices." << endl;
    ikVertex = new IK((int)IKVertexIDs.size(), IKVertexIDs.data(), fk, skinning, /*tag*/ 2);
  }

  // Default to joint mode if available; otherwise start in vertex mode.
  useVertexHandles = (ikJoint == nullptr);
  ik = useVertexHandles ? ikVertex : ikJoint;

  size_t numHandles = useVertexHandles ? IKVertexIDs.size() : IKJointIDs.size();
  IKJointPos.resize(numHandles);
  // We need an initial skinned mesh so vertex-handle world positions are correct.
  updateSkinnedMesh();
  for(size_t i = 0; i < numHandles; i++)
  {
    IKJointPos[i] = getHandleWorldPosition(i);
  }

  // ---------------------------------------------------
  // rendering setup
  // ---------------------------------------------------

  double cameraUp[3] = {0,1,0};

  Vec3d cameraFocus;
  Vec3d bmin, bmax;
  mesh->computeBoundingBox();
  mesh->getCubicBoundingBox(1.0, &bmin, &bmax);
  modelCenter = (bmin + bmax) / 2.0;
  modelRadius = mesh->getDiameter() / 2;

  // Tune IK sub-stepping relative to the model scale so the same driver works for
  // armadillo / dragon / hand without per-demo tweaks.
  double ikStepDistance = modelRadius * 0.05;
  if (ikJoint)  ikJoint->setSubStepping(20, ikStepDistance);
  if (ikVertex) ikVertex->setSubStepping(20, ikStepDistance);

  // compute the size of the shape for getting a proper renderedLocalAxisLength
  double localAxisLength = modelRadius / 5.0;
  skeletonRenderer = new SkeletonRenderer(fk, localAxisLength);
  cout << "Finished joint initialization" << endl;

  double cameraRadius = 0;
  cameraFocus = modelCenter;
  cameraRadius = modelRadius * 2.5;
  zNear = cameraRadius * 0.01;
  zFar = cameraRadius * 100.0;

  double cameraPhi = 270.0;
  double cameraTheta = 0;
  camera = new SphericalCamera(cameraRadius,
      1.0 * cameraPhi / 360 * (2.0 * PI),
      1.0 * cameraTheta / 360 * (2.0 * PI),
      &cameraFocus[0], cameraUp, 0.05);

  //  lighting = new Lighting;
  //  lighting->SetLightBox(&bmin[0], &bmax[0]);
  //  lighting->SetAllLightsEnabled(false);
  //  lighting->SetLightEnabled(2, true);
  //  lighting->SetLightEnabled(3, true);
  //  lighting->SetLightEnabled(6, true);
  //  lighting->SetLightEnabled(7, true);
  //  lighting->SetAllLightsIntensity(allLightsIntensity);

  cameraLighting = new CameraLighting;
  cameraLighting->SetLightIntensity(allLightsIntensity);

  //   clear to white
  glClearColor(256.0 / 256, 256.0 / 256, 256.0 / 256, 0.0);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_STENCIL_TEST);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_POLYGON_SMOOTH);
  glEnable(GL_LINE_SMOOTH);
  printf ("Initialization complete.\n");
  return;
}

#define ADD_CONFIG(v) configFile.addOptionOptional(#v, &v, v)
static void initConfigurations()
{
  ConfigFile configFile;

  ADD_CONFIG(allLightsIntensity);
  ADD_CONFIG(screenshotBaseName);
  ADD_CONFIG(meshFilename);

  // Maya data needs jointHierarchyFilename, jointRestTransformsFilename and jointWeightsFilename
  ADD_CONFIG(jointHierarchyFilename);
  ADD_CONFIG(jointRestTransformsFilename);
  ADD_CONFIG(jointWeightsFilename);
  ADD_CONFIG(IKJointIDs);
  ADD_CONFIG(IKVertexIDs);

  // parse the configuration file
  if (configFile.parseOptions(configFilename.c_str()) != 0)
  {
    printf("Error parsing options.\n");
    exit(1);
  }

  // The config variables have now been loaded with their specified values.
  // Informatively print the variables (with assigned values) that were just parsed.
  configFile.printOptions();
}

int main (int argc, char ** argv)
{
  int numFixedArgs = 2;
  if ( argc < numFixedArgs )
  {
    cout << "Renders an obj mesh on the screen." << endl;
    cout << "Usage: " << argv[0] << " configFilename" << endl;
    return 0;
  }
 
  configFilename = argv[1];

  initConfigurations();

  glutInit(&argc,argv);
  glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL | GLUT_MULTISAMPLE);

  windowWidth = 800;
  windowHeight = 800;

  glutInitWindowSize (windowWidth,windowHeight);
  glutInitWindowPosition (0,0);
  windowID = glutCreateWindow ("IK viewer");
  if (fullScreen==1)
    glutFullScreen();

  #ifdef __APPLE__
    // This is needed on recent Mac OS X versions to correctly display the window.
    glutReshapeWindow(windowWidth - 1, windowHeight - 1);
  #endif

  initialize();

  // callbacks
  glutDisplayFunc(displayFunction);
  glutMotionFunc(mouseDrag);
  glutPassiveMotionFunc(mouseNoDrag);
  glutIdleFunc(idleFunction);
  glutKeyboardFunc(keyboardFunc);
  glutSpecialFunc(specialKeysFunc);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouseButtonActivity);

  reshape(windowWidth,windowHeight);
  glutMainLoop();

  return(0);
}

