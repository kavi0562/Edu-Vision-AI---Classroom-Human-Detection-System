def count_people(results, conf_threshold=0.5):
    boxes = results[0].boxes
    count = 0
    total_conf = 0.0

    for box in boxes:
        if int(box.cls[0]) == 0 and box.conf[0] > conf_threshold:
            count += 1
            total_conf += float(box.conf[0])

    avg_conf = (total_conf / count) if count > 0 else 0.0
    return count, avg_conf