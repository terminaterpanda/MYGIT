from django import forms
from .models import TOKENIZER

class TokenizerForm(forms.ModelForm):
    class Meta:
        model = TOKENIZER
        fields = ['file']
        widgets = {
            'file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }



