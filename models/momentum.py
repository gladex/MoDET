import torch
import torch.nn.functional as F

def knowledge_distillation_loss(output_student, output_teacher, temperature):
    epsilon = 1e-10  
    student_probs = F.softmax(output_student / temperature, dim=1)
    teacher_probs = F.softmax(output_teacher / temperature, dim=1)
    row_max_values = torch.max(teacher_probs, dim=1)[0]
    max_value = torch.max(row_max_values)
    threshold = 0.7 * max_value  
    print(f"Max value: {max_value:.4f}, Threshold: {threshold:.4f}")
    rows_greater_than_threshold_indices = torch.where(row_max_values > threshold)[0]
    teacher_result = teacher_probs.clone()
    for idx in rows_greater_than_threshold_indices:
        max_idx = torch.argmax(teacher_probs[idx])
        teacher_result[idx] = torch.zeros_like(teacher_result[idx])
        teacher_result[idx, max_idx] = 1
    student_probs = student_probs + epsilon
    teacher_result = teacher_result + epsilon
    kl_div_loss = F.kl_div(student_probs.log(), teacher_result, reduction='batchmean') * (temperature ** 2)
    return kl_div_loss
