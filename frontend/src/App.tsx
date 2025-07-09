import React from 'react';
import { AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemText, CssBaseline, Box } from '@mui/material';
import Dashboard from './pages/Dashboard';
import Strategies from './pages/Strategies';
import Accounts from './pages/Accounts';
import Trades from './pages/Trades';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

const drawerWidth = 220;
const sections = [
  'Dashboard',
  'Strategies',
  'Accounts',
  'Trades',
  'Analytics',
  'Settings',
];

function App() {
  const [selected, setSelected] = React.useState('Dashboard');

  const renderContent = () => {
    switch (selected) {
      case 'Dashboard':
        return <Dashboard />;
      case 'Strategies':
        return <Strategies />;
      case 'Accounts':
        return <Accounts />;
      case 'Trades':
        return <Trades />;
      case 'Analytics':
        return <Analytics />;
      case 'Settings':
        return <Settings />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar position="fixed" sx={{ zIndex: 1201 }}>
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Trading Bot Dashboard
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {sections.map((text) => (
              <ListItem button key={text} selected={selected === text} onClick={() => setSelected(text)}>
                <ListItemText primary={text} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3, ml: `${drawerWidth}px` }}>
        <Toolbar />
        {renderContent()}
      </Box>
    </Box>
  );
}

export default App;
