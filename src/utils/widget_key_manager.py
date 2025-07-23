class WidgetKeyManager:
    """
    Manages unique keys for Streamlit widgets to prevent duplicate ID errors.
    """
    
    def __init__(self):
        """Initialize the key manager."""
        self.component_prefixes = {}
        self.key_counters = {}
    
    def register_component(self, component_name, prefix=None):
        """
        Registers a component for key generation.
        
        Args:
            component_name: Name of the component
            prefix: Optional prefix for keys (defaults to component_name)
        """
        self.component_prefixes[component_name] = prefix or component_name
        self.key_counters[component_name] = 0
    
    def get_key(self, component_name, widget_name):
        """
        Gets a unique key for a widget.
        
        Args:
            component_name: Name of the component
            widget_name: Name of the widget
            
        Returns:
            Unique key string
        """
        # Register component if not already registered
        if component_name not in self.component_prefixes:
            self.register_component(component_name)
            
        # Increment counter for this component
        self.key_counters[component_name] += 1
        
        # Create unique key
        prefix = self.component_prefixes[component_name]
        counter = self.key_counters[component_name]
        
        return f"{prefix}_{widget_name}_{counter}"
    
    def reset_component(self, component_name):
        """
        Resets the counter for a component.
        
        Args:
            component_name: Name of the component
        """
        if component_name in self.key_counters:
            self.key_counters[component_name] = 0 