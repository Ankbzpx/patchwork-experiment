#version 330 core
layout(location = 0) out vec4 fragColor;

uniform int iNum;
uniform vec4 fCoeffs[256];
uniform int iEnable[256];
uniform float fBeta;
uniform bool iSoftmax;
uniform bool iCandidate;
uniform bool iPadEps;
uniform float fLineScale;

in vec2 uv;

void main() {
    vec4 lightColor = vec4(0.50980, 0.73333, 0.89020, 1.0);
    vec4 darkColor = vec4(0.0, 0.18039, 0.29804, 1.0);
    vec4 lineColor = vec4(0.90196, 0.90196, 0.90196, 1.0);
    vec4 activateColor = vec4(0.94902, 0.55686, 0.16863, 1.0);
    float eps = 0.0001;

    if(iNum == 0) {
        fragColor = lightColor;
        return;
    }

    vec3 uv_homo = vec3(2.0 * (uv - 0.5), 1.0);

    bool has_pos = false, has_neg = false;
    int idx_pos = 0, idx_neg = 0;
    float max_val_pos = -1e20, max_val_neg = -1e20;

    int idx_max = 0, idx_second = 0;
    float max_val = -1e20, second_max_val = -1e20;

    if(iPadEps) {
        has_pos = true;
        max_val_pos = eps;
    }

    for(int i = 0; i < iNum; i++) {
        if(iEnable[i] == 0) {
            continue;
        }

        float val = dot(fCoeffs[i].xyz, uv_homo);
        if(fCoeffs[i].w >= 0.0) {
            has_pos = true;
            if(max_val_pos < val) {
                max_val_pos = val;
                idx_pos = i;
            }
        } else {
            has_neg = true;
            if(max_val_neg < val) {
                max_val_neg = val;
                idx_neg = i;
            }
        }

        if(max_val < val) {
            second_max_val = max_val;
            idx_second = idx_max;
            max_val = val;
            idx_max = i;
        } else if(second_max_val < val) {
            second_max_val = val;
            idx_second = i;
        }
    }
    float val_acc_pos = 0.0, val_acc_neg = 0.0;
    if(iSoftmax) {
        for(int i = 0; i < iNum; i++) {
            if(iEnable[i] == 0) {
                continue;
            }

            float val = dot(fCoeffs[i].xyz, uv_homo);
            if(fCoeffs[i].w >= 0.0) {
                val = fBeta * (val - max_val_pos);
                val_acc_pos += fCoeffs[i].w * exp(val);
            } else {
                val = fBeta * (val - max_val_neg);
                val_acc_neg += (-fCoeffs[i].w) * exp(val);
            }
        }
    }
    if(iPadEps) {
        val_acc_pos += exp(fBeta * (eps - max_val_pos));
    }
    if(!has_pos) {
        max_val_pos = 0.0;
        val_acc_pos = 1.0;
    }
    if(!has_neg) {
        max_val_neg = 0.0;
        val_acc_neg = 1.0;
    }
    float sdf_max = (max_val_pos - max_val_neg);
    if(iSoftmax) {
        sdf_max += (log(val_acc_pos) - log(val_acc_neg)) / fBeta;
    }
    fragColor = (sdf_max < 0.0) ? lightColor : darkColor;

    if(iCandidate) {
        float line_scale = length(fCoeffs[idx_max].xy - fCoeffs[idx_second].xy);
        if(abs(max_val - second_max_val) / line_scale < fLineScale * 0.01) {
            fragColor = lineColor;
        }
    }

    // Activate segment
    if(has_pos && has_neg) {
        float sdf_scale = length(fCoeffs[idx_pos].xy - fCoeffs[idx_neg].xy);
        if((abs(sdf_max) / sdf_scale) < fLineScale * 0.01) {
            fragColor = activateColor;
        }
    }
}