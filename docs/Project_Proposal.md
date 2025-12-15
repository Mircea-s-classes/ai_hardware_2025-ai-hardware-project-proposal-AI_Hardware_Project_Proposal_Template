# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal Template

## 1. Project Title
Ring Camera with AI

Nate Owen, Sammie Levine, Grayson Turner, Marissa Cash

Identifing specfic humans using facial recognition with Google Coral

## 2. Platform Selection
Google Coral, Edge AI

## 3. Problem Definition
Home owners that use Ring cameras want the ability to accurately identify know persons from unknown persons that are at their front door. Our group plans to emulate the function of a ring camera by creating an AI program to identify unique faces. This will allow homeowners to be alerted if strangers are at their door. This project is relvant to AI hardware as it incorporates the importance of latency by quickly identify faces and effeciency by correctly identifying known faces from unknown strangers.

## 4. Technical Objectives
- Use our model to accuratly identify 3 unique faces with an above 80 success rate in normal lighting conditions
- Maintain an average latency of 500ms from image capture to recognition
- Maintain a false postive postive rate of less than 2 percent

## 5. Methodology
1) First, we want to use Edge Impulse Studio to train our model and then be able to export it to a TFLite model.
2) Compile this TFlite model to then be deployed onto our Google Coral Edge TPU.
3) By plugging in our Coral USB Accelerator, we can then use our laptop's webcam with a pyhton file to install Coral's libraries.
4) Then we can communicate with the laptop and the TPU to send image data and have the TPU send the results back.
5) After this, we can measure our quantitative data based on our three points of focus. 

## 6. Expected Deliverables
List tangible outputs: 
- Working demo:
    Working demo using Google Coral hardware
- GitHub repository:
    All software created and maintained in the Github Repo
- Documentation:
    Neccesary documentation to run and excute the software
- Presentation slides
    Completed presentation slides by all group members
- Final report:
    Completed final report by all group members

## 7. Team Responsibilities

| Name | Role | Responsibilities |
|------|------|------------------|
| Grayson Turner | Team Lead | Coordination, documentation |
| Nate Owen | Hardware | Setup, integration |
| Sammie Levine | Software | Model training, inference |
| Marissa Cash | Evaluation | Testing, benchmarking |

## 8. Timeline and Milestones
Provide expected milestones:

| Week | Milestone | Deliverable |
|------|------------|-------------|
| Nov. 5/6 | Proposal | PDF + GitHub submission |
| 4 | Midterm presentation | Slides, preliminary results |
| 5 | Integration & testing | Working prototype |
| Mid December | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
List special hardware, datasets, or compute access needed.
https://www.kaggle.com/datasets/atulanandjha/lfwpeople

## 10. References
Include relevant papers, repositories, and documentation.
https://www.coral.ai/examples/
https://embecosm.com/2019/08/09/facial-recognition-on-the-new-google-coral-development-board/
