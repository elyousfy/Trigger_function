curr_faces = len(result)
                if curr_faces > face_counter_global:
                    time.sleep(0.25)
                    ret, frame = cam.read()
                    results = model(frame)
                    result = results[0]